// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// End-to-end convolutional network on the emulated multi-core NPU.
//
// Runs a small but complete CNN classifier using only the device/runtime API
// (device.h) and the C standard library -- no IREE, no PyTorch, no Python:
//
//   conv1 : [B,1,16,16] --3x3 pad1, +bias, ReLU--> [B,8,16,16]
//   pool1 : max 2x2 stride 2                     -> [B,8,8,8]
//   conv2 : 3x3 pad1, +bias, ReLU                -> [B,16,8,8]
//   pool2 : avg 2x2 stride 2                     -> [B,16,4,4]
//   fc    : flatten [B,256] --Linear 256->10, +bias--> logits [B,10]
//   out   : softmax over the 10 classes, then argmax -> predicted class
//
// Every stage runs on the device: the convolutions lower to im2col + GEMM on
// the 16x16 tensor cores, ReLU runs on the activation unit, pooling and softmax
// on the vector unit, and the argmax is a device reduction. The whole thing is
// validated against a host reference computed in double precision.
//
//   # watch the cores while it runs (two terminals):
//   python demo/visualizer.py /tmp/emunpu.events
//   IREE_EMUNPU_VIZ=/tmp/emunpu.events ./cnn_demo
//
// IREE_EMUNPU_TRACE=1 logs every instruction the cores execute.
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "emunpu/device.h"

enum {
  B = 4,                // batch
  H0 = 16,              // input height/width
  C0 = 1,               // input channels
  C1 = 8,               // conv1 output channels
  C2 = 16,              // conv2 output channels
  H1 = H0 / 2,          // after pool1: 8
  H2 = H1 / 2,          // after pool2: 4
  FLAT = C2 * H2 * H2,  // 256
  CLASSES = 10,
};

#define CHECK_OK(expr)                                                      \
  do {                                                                      \
    emunpu_status_t _s = (expr);                                            \
    if (_s != EMUNPU_OK) {                                                  \
      fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, \
              emunpu_status_name(_s));                                      \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// Deterministic pseudo-random values in ~[-0.5, 0.5).
static void fill(float* p, size_t n, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    p[i] = (float)((int)((s >> 16) & 0xff) - 128) * (1.0f / 256.0f);
  }
}

//===----------------------------------------------------------------------===//
// Host reference (double precision, written straight from the definitions)
//===----------------------------------------------------------------------===//

// NCHW convolution with 'pad' zero padding, stride 1, bias and optional ReLU.
static void ref_conv3x3(const double* in, const float* w, const float* bias,
                        double* out, uint32_t batch, uint32_t in_c,
                        uint32_t out_c, uint32_t hw, int relu) {
  const uint32_t k = 3, pad = 1;
  for (uint32_t b = 0; b < batch; ++b) {
    for (uint32_t oc = 0; oc < out_c; ++oc) {
      for (uint32_t oy = 0; oy < hw; ++oy) {
        for (uint32_t ox = 0; ox < hw; ++ox) {
          double acc = bias[oc];
          for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < k; ++ky) {
              long iy = (long)oy - pad + ky;
              if (iy < 0 || iy >= (long)hw) continue;
              for (uint32_t kx = 0; kx < k; ++kx) {
                long ix = (long)ox - pad + kx;
                if (ix < 0 || ix >= (long)hw) continue;
                acc += in[((b * in_c + ic) * hw + iy) * hw + ix] *
                       w[((oc * in_c + ic) * k + ky) * k + kx];
              }
            }
          }
          if (relu && acc < 0.0) acc = 0.0;
          out[((b * out_c + oc) * hw + oy) * hw + ox] = acc;
        }
      }
    }
  }
}

// 2x2 stride-2 pooling (no padding, so every window is full).
static void ref_pool2x2(const double* in, double* out, uint32_t planes,
                        uint32_t in_hw, int is_max) {
  uint32_t out_hw = in_hw / 2;
  for (uint32_t p = 0; p < planes; ++p) {
    for (uint32_t oy = 0; oy < out_hw; ++oy) {
      for (uint32_t ox = 0; ox < out_hw; ++ox) {
        double best = -INFINITY, sum = 0.0;
        for (uint32_t ky = 0; ky < 2; ++ky) {
          for (uint32_t kx = 0; kx < 2; ++kx) {
            double v = in[(p * in_hw + oy * 2 + ky) * in_hw + ox * 2 + kx];
            if (v > best) best = v;
            sum += v;
          }
        }
        out[(p * out_hw + oy) * out_hw + ox] = is_max ? best : sum / 4.0;
      }
    }
  }
}

int main(void) {
  printf("emunpu CNN demo:  x[%d,%d,%d,%d]\n", B, C0, H0, H0);
  printf(
      "  conv3x3 %d->%d +relu | maxpool2 | conv3x3 %d->%d +relu | avgpool2"
      " | linear %d->%d | softmax\n",
      C0, C1, C1, C2, FLAT, CLASSES);

  emunpu_device_t* dev = NULL;
  CHECK_OK(emunpu_device_create(0, &dev));
  printf("device: %u cores\n\n", emunpu_device_num_cores(dev));

  // --- Host-side parameters ---------------------------------------------
  const size_t x_elems = (size_t)B * C0 * H0 * H0;
  const size_t w1_elems = (size_t)C1 * C0 * 9;
  const size_t w2_elems = (size_t)C2 * C1 * 9;
  const size_t wfc_elems = (size_t)CLASSES * FLAT;
  float* x = malloc(sizeof(float) * x_elems);
  float* w1 = malloc(sizeof(float) * w1_elems);
  float* b1 = malloc(sizeof(float) * C1);
  float* w2 = malloc(sizeof(float) * w2_elems);
  float* b2 = malloc(sizeof(float) * C2);
  float* wfc = malloc(sizeof(float) * wfc_elems);
  float* bfc = malloc(sizeof(float) * CLASSES);
  fill(x, x_elems, 1);
  fill(w1, w1_elems, 2);
  fill(b1, C1, 3);
  fill(w2, w2_elems, 4);
  fill(b2, C2, 5);
  fill(wfc, wfc_elems, 6);
  fill(bfc, CLASSES, 7);

  // --- Device tensors ---------------------------------------------------
  const emunpu_dtype_t f32 = EMUNPU_DT_FP32;
  emunpu_tensor_id_t d_x, d_w1, d_b1, d_h1, d_p1, d_w2, d_b2, d_h2, d_p2;
  emunpu_tensor_id_t d_wfc, d_bfc, d_logits, d_probs, d_pred;
  uint32_t n_x = (uint32_t)x_elems, n_w1 = (uint32_t)w1_elems;
  uint32_t n_w2 = (uint32_t)w2_elems, n_wfc = (uint32_t)wfc_elems;
  uint32_t n_h1 = B * C1 * H0 * H0, n_p1 = B * C1 * H1 * H1;
  uint32_t n_h2 = B * C2 * H1 * H1, n_p2 = B * C2 * H2 * H2;
  uint32_t n_out = B * CLASSES, n_pred = B;
  uint32_t s_c1 = C1, s_c2 = C2, s_cls = CLASSES;
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_x, "x", &d_x));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_w1, "w1", &d_w1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &s_c1, "b1", &d_b1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_h1, "h1", &d_h1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_p1, "p1", &d_p1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_w2, "w2", &d_w2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &s_c2, "b2", &d_b2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_h2, "h2", &d_h2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_p2, "p2", &d_p2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_wfc, "wfc", &d_wfc));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &s_cls, "bfc", &d_bfc));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_out, "logits", &d_logits));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_out, "probs", &d_probs));
  CHECK_OK(
      emunpu_device_alloc(dev, EMUNPU_DT_INT32, 1, &n_pred, "pred", &d_pred));

  CHECK_OK(emunpu_device_upload(dev, d_x, x, sizeof(float) * x_elems));
  CHECK_OK(emunpu_device_upload(dev, d_w1, w1, sizeof(float) * w1_elems));
  CHECK_OK(emunpu_device_upload(dev, d_b1, b1, sizeof(float) * C1));
  CHECK_OK(emunpu_device_upload(dev, d_w2, w2, sizeof(float) * w2_elems));
  CHECK_OK(emunpu_device_upload(dev, d_b2, b2, sizeof(float) * C2));
  CHECK_OK(emunpu_device_upload(dev, d_wfc, wfc, sizeof(float) * wfc_elems));
  CHECK_OK(emunpu_device_upload(dev, d_bfc, bfc, sizeof(float) * CLASSES));

  // --- The network, on the device ---------------------------------------
  const emunpu_act_t relu = {EMUNPU_ACT_RELU, 0.0, 0.0};

  emunpu_window_t conv1 = {0};
  conv1.batch = B;
  conv1.channels = C0;
  conv1.in_h = conv1.in_w = H0;
  conv1.kernel_h = conv1.kernel_w = 3;
  conv1.stride_h = conv1.stride_w = 1;
  conv1.pad_h = conv1.pad_w = 1;
  CHECK_OK(emunpu_conv2d(dev, NULL, d_h1, d_x, d_w1, d_b1, &conv1, C1, &relu));

  emunpu_window_t pool1 = {0};
  pool1.batch = B;
  pool1.channels = C1;
  pool1.in_h = pool1.in_w = H0;
  pool1.kernel_h = pool1.kernel_w = 2;
  pool1.stride_h = pool1.stride_w = 2;
  CHECK_OK(emunpu_pool2d(dev, NULL, d_p1, d_h1, &pool1, EMUNPU_POOL_MAX));

  emunpu_window_t conv2 = {0};
  conv2.batch = B;
  conv2.channels = C1;
  conv2.in_h = conv2.in_w = H1;
  conv2.kernel_h = conv2.kernel_w = 3;
  conv2.stride_h = conv2.stride_w = 1;
  conv2.pad_h = conv2.pad_w = 1;
  CHECK_OK(emunpu_conv2d(dev, NULL, d_h2, d_p1, d_w2, d_b2, &conv2, C2, &relu));

  emunpu_window_t pool2 = {0};
  pool2.batch = B;
  pool2.channels = C2;
  pool2.in_h = pool2.in_w = H1;
  pool2.kernel_h = pool2.kernel_w = 2;
  pool2.stride_h = pool2.stride_w = 2;
  CHECK_OK(emunpu_pool2d(dev, NULL, d_p2, d_h2, &pool2, EMUNPU_POOL_AVG));

  // pool2's [B,C2,H2,H2] output is already contiguous, so the "flatten" to
  // [B, FLAT] is a reinterpretation with no data movement.
  CHECK_OK(emunpu_linear(dev, NULL, d_logits, d_p2, d_wfc, d_bfc, B, CLASSES,
                         FLAT, /*transpose_b=*/1, /*act=*/NULL));
  CHECK_OK(emunpu_softmax(dev, NULL, d_probs, d_logits, B, CLASSES,
                          /*log_softmax=*/0));
  CHECK_OK(emunpu_reduce(dev, NULL, d_pred, d_probs, B, CLASSES,
                         EMUNPU_REDUCE_ARGMAX));
  CHECK_OK(emunpu_device_synchronize(dev));

  float* probs = malloc(sizeof(float) * n_out);
  int32_t pred[B];
  CHECK_OK(emunpu_device_download(dev, d_probs, probs, sizeof(float) * n_out));
  CHECK_OK(emunpu_device_download(dev, d_pred, pred, sizeof(pred)));

  // --- Host reference ---------------------------------------------------
  double* r_x = malloc(sizeof(double) * x_elems);
  for (size_t i = 0; i < x_elems; ++i) r_x[i] = x[i];
  double* r_h1 = malloc(sizeof(double) * n_h1);
  double* r_p1 = malloc(sizeof(double) * n_p1);
  double* r_h2 = malloc(sizeof(double) * n_h2);
  double* r_p2 = malloc(sizeof(double) * n_p2);
  double* r_probs = malloc(sizeof(double) * n_out);
  ref_conv3x3(r_x, w1, b1, r_h1, B, C0, C1, H0, /*relu=*/1);
  ref_pool2x2(r_h1, r_p1, B * C1, H0, /*is_max=*/1);
  ref_conv3x3(r_p1, w2, b2, r_h2, B, C1, C2, H1, /*relu=*/1);
  ref_pool2x2(r_h2, r_p2, B * C2, H1, /*is_max=*/0);
  for (uint32_t b = 0; b < B; ++b) {
    double logits[CLASSES], max = -INFINITY, sum = 0.0;
    for (uint32_t c = 0; c < CLASSES; ++c) {
      double acc = bfc[c];
      for (uint32_t j = 0; j < FLAT; ++j) {
        acc += r_p2[(size_t)b * FLAT + j] * wfc[(size_t)c * FLAT + j];
      }
      logits[c] = acc;
      if (acc > max) max = acc;
    }
    for (uint32_t c = 0; c < CLASSES; ++c) sum += exp(logits[c] - max);
    for (uint32_t c = 0; c < CLASSES; ++c) {
      r_probs[(size_t)b * CLASSES + c] = exp(logits[c] - max) / sum;
    }
  }

  // --- Report -----------------------------------------------------------
  double max_abs_err = 0.0;
  for (uint32_t i = 0; i < n_out; ++i) {
    double e = fabs((double)probs[i] - r_probs[i]);
    if (e > max_abs_err) max_abs_err = e;
  }
  int class_ok = 1;
  for (uint32_t b = 0; b < B; ++b) {
    uint32_t best = 0;
    for (uint32_t c = 1; c < CLASSES; ++c) {
      if (r_probs[(size_t)b * CLASSES + c] >
          r_probs[(size_t)b * CLASSES + best])
        best = c;
    }
    if (pred[b] != (int32_t)best) class_ok = 0;
  }

  for (uint32_t b = 0; b < B; ++b) {
    printf("sample %u: class %d  p=%.4f   probs[0:5] =", b, pred[b],
           probs[(size_t)b * CLASSES + pred[b]]);
    for (uint32_t c = 0; c < 5; ++c)
      printf(" %.4f", probs[(size_t)b * CLASSES + c]);
    printf("\n");
  }
  int pass = max_abs_err < 1e-5 && class_ok;
  printf("\nmax abs error vs host reference: %.3e\n", max_abs_err);
  printf("device argmax matches host: %s\n", class_ok ? "yes" : "NO");
  printf("-> %s\n", pass ? "PASS" : "FAIL");

  free(x);
  free(w1);
  free(b1);
  free(w2);
  free(b2);
  free(wfc);
  free(bfc);
  free(probs);
  free(r_x);
  free(r_h1);
  free(r_p1);
  free(r_h2);
  free(r_p2);
  free(r_probs);
  emunpu_device_destroy(dev);
  return pass ? 0 : 1;
}
