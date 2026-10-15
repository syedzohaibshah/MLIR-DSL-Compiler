// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Standalone end-to-end demo for the emulated multi-core NPU.
//
// Runs a tiny 2-layer MLP entirely on the simulated device, using nothing but
// the device/runtime API (device.h) and the C standard library -- no
// IREE, no PyTorch, no Python. It is the C analogue of the iree-turbine PyTorch
// demos, meant as a copy-paste starting point for wiring the accelerator into
// your own compiler/runtime stack.
//
//   layer1:  h = relu(x @ W1^T + b1)      x:[B,IN]  W1:[H,IN]  b1:[H]
//   layer2:  y =      h @ W2^T + b2        W2:[OUT,H]          b2:[OUT]
//
// The weights are stored transposed (row-major [out, in]) exactly like
// torch.nn.Linear, so the matmuls use transpose_b=1. Both layers are larger
// than the 16x16 tensor-core tile, so the scheduler fans the work across all
// IREE_EMUNPU_NUM_CORES cores -- run with the visualizer attached to watch it:
//
//   # terminal 1
//   python demo/visualizer.py /tmp/emunpu.events
//   # terminal 2
//   IREE_EMUNPU_VIZ=/tmp/emunpu.events ./mlp_demo
//
// Set IREE_EMUNPU_TRACE=1 to log every op the engine executes.
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "emunpu/device.h"

// Problem sizes. All dims > 16 so every matmul tiles across the cores.
enum { B = 64, IN = 128, H = 256, OUT = 32 };

#define CHECK_OK(expr)                                                      \
  do {                                                                      \
    emunpu_status_t _s = (expr);                                            \
    if (_s != EMUNPU_OK) {                                                  \
      fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, \
              emunpu_status_name(_s));                                      \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// Deterministic pseudo-random weights/inputs in a small range so fp32 results
// are easy to sanity-check and reproducible run to run.
static void fill(float* p, size_t n, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    p[i] = (float)((int)((s >> 16) & 0xff) - 128) *
           (1.0f / 256.0f);  // ~[-0.5,0.5)
  }
}

// Host reference for one linear+bias(+relu) layer, to validate the device.
static void ref_linear(const float* x, const float* w, const float* bias,
                       float* out, int m, int n, int k, int relu) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = bias ? bias[j] : 0.0;
      for (int p = 0; p < k; ++p) acc += (double)x[i * k + p] * w[j * k + p];
      out[i * n + j] = relu && acc < 0.0 ? 0.0f : (float)acc;
    }
  }
}

int main(void) {
  printf("emunpu MLP demo:  x[%d,%d] -> relu(Linear %d) -> Linear %d\n", B, IN,
         H, OUT);

  emunpu_device_t* dev = NULL;
  CHECK_OK(emunpu_device_create(0, &dev));
  printf("device: %u cores\n\n", emunpu_device_num_cores(dev));

  // --- Host tensors -----------------------------------------------------
  float* x = malloc(sizeof(float) * B * IN);
  float* W1 = malloc(sizeof(float) * H * IN);
  float* b1 = malloc(sizeof(float) * H);
  float* W2 = malloc(sizeof(float) * OUT * H);
  float* b2 = malloc(sizeof(float) * OUT);
  float* y = malloc(sizeof(float) * B * OUT);
  fill(x, (size_t)B * IN, 1);
  fill(W1, (size_t)H * IN, 2);
  fill(b1, H, 3);
  fill(W2, (size_t)OUT * H, 4);
  fill(b2, OUT, 5);

  // --- Device tensors (allocated in global memory) ----------------------
  const emunpu_dtype_t f32 = EMUNPU_DT_FP32;
  emunpu_tensor_id_t dx, dW1, db1, dh, dW2, db2, dy;
  uint32_t s_x[2] = {B, IN}, s_W1[2] = {H, IN}, s_b1 = H, s_h[2] = {B, H};
  uint32_t s_W2[2] = {OUT, H}, s_b2 = OUT, s_y[2] = {B, OUT};
  CHECK_OK(emunpu_device_alloc(dev, f32, 2, s_x, "x", &dx));
  CHECK_OK(emunpu_device_alloc(dev, f32, 2, s_W1, "W1", &dW1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &s_b1, "b1", &db1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 2, s_h, "h", &dh));
  CHECK_OK(emunpu_device_alloc(dev, f32, 2, s_W2, "W2", &dW2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &s_b2, "b2", &db2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 2, s_y, "y", &dy));

  // --- Upload host -> device --------------------------------------------
  CHECK_OK(emunpu_device_upload(dev, dx, x, sizeof(float) * B * IN));
  CHECK_OK(emunpu_device_upload(dev, dW1, W1, sizeof(float) * H * IN));
  CHECK_OK(emunpu_device_upload(dev, db1, b1, sizeof(float) * H));
  CHECK_OK(emunpu_device_upload(dev, dW2, W2, sizeof(float) * OUT * H));
  CHECK_OK(emunpu_device_upload(dev, db2, b2, sizeof(float) * OUT));

  // --- Compute on the NPU -----------------------------------------------
  // layer1: h = relu(x @ W1^T + b1)  (transpose_b=1, fuse_relu=1)
  CHECK_OK(emunpu_gemm(dev, /*stream=*/NULL, dh, dx, dW1, db1, B, H, IN,
                       /*transpose_b=*/1, /*fuse_relu=*/1));
  // layer2: y = h @ W2^T + b2        (no relu)
  CHECK_OK(emunpu_gemm(dev, /*stream=*/NULL, dy, dh, dW2, db2, B, OUT, H,
                       /*transpose_b=*/1, /*fuse_relu=*/0));
  CHECK_OK(emunpu_device_synchronize(dev));

  // --- Download result --------------------------------------------------
  CHECK_OK(emunpu_device_download(dev, dy, y, sizeof(float) * B * OUT));

  // --- Validate against a host reference --------------------------------
  float* hh = malloc(sizeof(float) * B * H);
  float* ry = malloc(sizeof(float) * B * OUT);
  ref_linear(x, W1, b1, hh, B, H, IN, /*relu=*/1);
  ref_linear(hh, W2, b2, ry, B, OUT, H, /*relu=*/0);
  double max_abs_err = 0.0;
  for (int i = 0; i < B * OUT; ++i) {
    double e = fabs((double)y[i] - (double)ry[i]);
    if (e > max_abs_err) max_abs_err = e;
  }

  printf("output y[0, 0:6] = ");
  for (int j = 0; j < 6 && j < OUT; ++j) printf("% .4f ", y[j]);
  printf("\nmax abs error vs host reference: %.3e  -> %s\n", max_abs_err,
         max_abs_err < 1e-3 ? "PASS" : "FAIL");

  free(x);
  free(W1);
  free(b1);
  free(W2);
  free(b2);
  free(y);
  free(hh);
  free(ry);
  emunpu_device_free(dev, dx);
  emunpu_device_free(dev, dW1);
  emunpu_device_free(dev, db1);
  emunpu_device_free(dev, dh);
  emunpu_device_free(dev, dW2);
  emunpu_device_free(dev, db2);
  emunpu_device_free(dev, dy);
  emunpu_device_destroy(dev);
  return max_abs_err < 1e-3 ? 0 : 1;
}
