// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// End-to-end transformer block on the emulated multi-core NPU.
//
// A complete pre-norm self-attention + feed-forward block, token embeddings and
// all, using only the device/runtime API (device.h) and the C standard library:
//
//   x     = embedding[tokens]                        gather      [T, D]
//   xn    = layernorm(x) * gamma + beta              norm + EW   [T, D]
//   Q,K,V = xn * Wq^T, xn * Wk^T, xn * Wv^T          3 x GEMM    [T, D]
//   S     = (Q * K^T) * (1/sqrt(D))                  transpose + GEMM + EW
//   A     = softmax(S)                               softmax     [T, T]
//   ctx   = A * V                                    GEMM        [T, D]
//   h     = x + ctx * Wo^T + bo                      GEMM + EW   [T, D]
//   hn    = layernorm(h) * gamma2 + beta2            norm + EW
//   y     = h + (gelu(hn * W1^T + b1)) * W2^T + b2   2 x Linear  [T, D]
//
// This exercises the operations a real attention block needs -- gather,
// layernorm, transpose, scaled matmul, row softmax, GELU, and residual adds --
// and validates the result against a host reference computed in double
// precision.
//
//   # watch the cores while it runs (two terminals):
//   python demo/visualizer.py /tmp/emunpu.events
//   IREE_EMUNPU_VIZ=/tmp/emunpu.events ./transformer_demo
//
// IREE_EMUNPU_TRACE=1 logs every instruction the cores execute.
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "emunpu/device.h"

enum {
  VOCAB = 64,  // embedding table rows
  T = 24,      // tokens (sequence length)
  D = 32,      // model width
  FF = 64,     // feed-forward hidden width
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

static void fill(float* p, size_t n, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    p[i] = (float)((int)((s >> 16) & 0xff) - 128) * (1.0f / 256.0f);
  }
}

//===----------------------------------------------------------------------===//
// Host reference helpers (double precision, straight from the definitions)
//===----------------------------------------------------------------------===//

// out[m,n] = sum_k a[m,k] * b[n,k] (+ bias[n]) -- i.e. a transpose_b matmul,
// the layout torch.nn.Linear stores its weight in.
static void ref_linear(const double* a, const float* b, const float* bias,
                       double* out, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      double acc = bias ? bias[n] : 0.0;
      for (uint32_t k = 0; k < K; ++k) acc += a[m * K + k] * b[n * K + k];
      out[m * N + n] = acc;
    }
  }
}

static void ref_layernorm(const double* in, const float* gamma,
                          const float* beta, double* out, uint32_t rows,
                          uint32_t width, double eps) {
  for (uint32_t r = 0; r < rows; ++r) {
    const double* row = in + (size_t)r * width;
    double mean = 0.0;
    for (uint32_t j = 0; j < width; ++j) mean += row[j];
    mean /= (double)width;
    double var = 0.0;
    for (uint32_t j = 0; j < width; ++j)
      var += (row[j] - mean) * (row[j] - mean);
    var /= (double)width;
    for (uint32_t j = 0; j < width; ++j) {
      out[(size_t)r * width + j] =
          (row[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j];
    }
  }
}

static void ref_softmax_rows(double* m, uint32_t rows, uint32_t width) {
  for (uint32_t r = 0; r < rows; ++r) {
    double* row = m + (size_t)r * width;
    double max = row[0];
    for (uint32_t j = 1; j < width; ++j)
      if (row[j] > max) max = row[j];
    double sum = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
      row[j] = exp(row[j] - max);
      sum += row[j];
    }
    for (uint32_t j = 0; j < width; ++j) row[j] /= sum;
  }
}

static double ref_gelu(double x) {
  return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

int main(void) {
  printf("emunpu transformer demo:  %d tokens x %d dims (ffn %d), vocab %d\n",
         T, D, FF, VOCAB);
  printf(
      "  gather | layernorm | q/k/v gemm | transpose | scaled attn |"
      " softmax | gemm | residual | layernorm | gelu ffn | residual\n");

  emunpu_device_t* dev = NULL;
  CHECK_OK(emunpu_device_create(0, &dev));
  printf("device: %u cores\n\n", emunpu_device_num_cores(dev));

  // --- Host-side parameters ---------------------------------------------
  float* emb = malloc(sizeof(float) * VOCAB * D);
  float* wq = malloc(sizeof(float) * D * D);
  float* wk = malloc(sizeof(float) * D * D);
  float* wv = malloc(sizeof(float) * D * D);
  float* wo = malloc(sizeof(float) * D * D);
  float* bo = malloc(sizeof(float) * D);
  float* w1 = malloc(sizeof(float) * FF * D);
  float* b1 = malloc(sizeof(float) * FF);
  float* w2 = malloc(sizeof(float) * D * FF);
  float* b2 = malloc(sizeof(float) * D);
  float g1[D], be1[D], g2[D], be2[D];
  int32_t tokens[T];
  fill(emb, (size_t)VOCAB * D, 1);
  fill(wq, (size_t)D * D, 2);
  fill(wk, (size_t)D * D, 3);
  fill(wv, (size_t)D * D, 4);
  fill(wo, (size_t)D * D, 5);
  fill(bo, D, 6);
  fill(w1, (size_t)FF * D, 7);
  fill(b1, FF, 8);
  fill(w2, (size_t)D * FF, 9);
  fill(b2, D, 10);
  for (uint32_t i = 0; i < D; ++i) {
    g1[i] = 1.0f + 0.05f * (float)(i % 5);
    be1[i] = 0.01f * (float)(i % 7);
    g2[i] = 0.9f + 0.03f * (float)(i % 4);
    be2[i] = -0.02f * (float)(i % 3);
  }
  for (uint32_t i = 0; i < T; ++i) tokens[i] = (int32_t)((i * 11 + 5) % VOCAB);

  const double eps = 1e-5;
  const float inv_sqrt_d = (float)(1.0 / sqrt((double)D));

  // --- Device tensors ---------------------------------------------------
  const emunpu_dtype_t f32 = EMUNPU_DT_FP32;
  emunpu_tensor_id_t d_emb, d_tok, d_x, d_xn, d_q, d_k, d_kt, d_v, d_s, d_ctx;
  emunpu_tensor_id_t d_h, d_hn, d_ff, d_y, d_scale;
  emunpu_tensor_id_t d_wq, d_wk, d_wv, d_wo, d_bo, d_w1, d_b1, d_w2, d_b2;
  emunpu_tensor_id_t d_g1, d_be1, d_g2, d_be2;
  uint32_t n_emb = VOCAB * D, n_tok = T, n_td = T * D, n_tt = T * T;
  uint32_t n_dd = D * D, n_d = D, n_ffd = FF * D, n_ff = FF, n_tff = T * FF;
  uint32_t one = 1;
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_emb, "emb", &d_emb));
  CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_INT32, 1, &n_tok, "tok", &d_tok));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "x", &d_x));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "xn", &d_xn));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "q", &d_q));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "k", &d_k));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "kt", &d_kt));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "v", &d_v));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_tt, "scores", &d_s));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "ctx", &d_ctx));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "h", &d_h));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "hn", &d_hn));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_tff, "ff", &d_ff));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_td, "y", &d_y));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &one, "scale", &d_scale));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_dd, "wq", &d_wq));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_dd, "wk", &d_wk));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_dd, "wv", &d_wv));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_dd, "wo", &d_wo));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "bo", &d_bo));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_ffd, "w1", &d_w1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_ff, "b1", &d_b1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_ffd, "w2", &d_w2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "b2", &d_b2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "g1", &d_g1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "be1", &d_be1));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "g2", &d_g2));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_d, "be2", &d_be2));

#define UPLOAD(id, host, count) \
  CHECK_OK(emunpu_device_upload(dev, id, host, sizeof(float) * (count)))
  UPLOAD(d_emb, emb, (size_t)VOCAB * D);
  UPLOAD(d_wq, wq, (size_t)D * D);
  UPLOAD(d_wk, wk, (size_t)D * D);
  UPLOAD(d_wv, wv, (size_t)D * D);
  UPLOAD(d_wo, wo, (size_t)D * D);
  UPLOAD(d_bo, bo, D);
  UPLOAD(d_w1, w1, (size_t)FF * D);
  UPLOAD(d_b1, b1, FF);
  UPLOAD(d_w2, w2, (size_t)D * FF);
  UPLOAD(d_b2, b2, D);
  UPLOAD(d_g1, g1, D);
  UPLOAD(d_be1, be1, D);
  UPLOAD(d_g2, g2, D);
  UPLOAD(d_be2, be2, D);
  UPLOAD(d_scale, &inv_sqrt_d, 1);
#undef UPLOAD
  CHECK_OK(emunpu_device_upload(dev, d_tok, tokens, sizeof(tokens)));

  // --- The block, on the device -----------------------------------------
  // 1. token embeddings: x = emb[tokens]
  CHECK_OK(emunpu_gather(dev, NULL, d_x, d_emb, d_tok, VOCAB, T, D));
  // 2. pre-norm
  CHECK_OK(emunpu_layernorm(dev, NULL, d_xn, d_x, d_g1, d_be1, T, D, eps));
  // 3. projections (weights are [D, D], so transpose_b=1)
  CHECK_OK(emunpu_matmul(dev, NULL, d_q, d_xn, d_wq, T, D, D, 1));
  CHECK_OK(emunpu_matmul(dev, NULL, d_k, d_xn, d_wk, T, D, D, 1));
  CHECK_OK(emunpu_matmul(dev, NULL, d_v, d_xn, d_wv, T, D, D, 1));
  // 4. scores = Q * K^T, via an explicit transpose + standard-layout matmul.
  CHECK_OK(emunpu_transpose(dev, NULL, d_kt, d_k, T, D));
  CHECK_OK(emunpu_matmul(dev, NULL, d_s, d_q, d_kt, T, T, D,
                         /*transpose_b=*/0));
  // 5. scale by 1/sqrt(D): a period-1 broadcast multiply by a scalar tensor.
  CHECK_OK(emunpu_vector_op(dev, NULL, d_s, d_s, d_scale, T * T, EMUNPU_EW_MUL,
                            /*bcast_period=*/1));
  // 6. attention weights
  CHECK_OK(emunpu_softmax(dev, NULL, d_s, d_s, T, T, /*log_softmax=*/0));
  // 7. ctx = A * V (A is [T,T], V is [T,D]: standard layout)
  CHECK_OK(emunpu_matmul(dev, NULL, d_ctx, d_s, d_v, T, D, T,
                         /*transpose_b=*/0));
  // 8. output projection + residual: h = x + (ctx * Wo^T + bo)
  CHECK_OK(emunpu_linear(dev, NULL, d_h, d_ctx, d_wo, d_bo, T, D, D, 1, NULL));
  CHECK_OK(emunpu_vector_op(dev, NULL, d_h, d_h, d_x, T * D, EMUNPU_EW_ADD, 0));
  // 9. feed-forward with a GELU epilogue, then the second residual.
  CHECK_OK(emunpu_layernorm(dev, NULL, d_hn, d_h, d_g2, d_be2, T, D, eps));
  const emunpu_act_t gelu = {EMUNPU_ACT_GELU, 0.0, 0.0};
  CHECK_OK(
      emunpu_linear(dev, NULL, d_ff, d_hn, d_w1, d_b1, T, FF, D, 1, &gelu));
  CHECK_OK(emunpu_linear(dev, NULL, d_y, d_ff, d_w2, d_b2, T, D, FF, 1, NULL));
  CHECK_OK(emunpu_vector_op(dev, NULL, d_y, d_y, d_h, T * D, EMUNPU_EW_ADD, 0));
  CHECK_OK(emunpu_device_synchronize(dev));

  float* y = malloc(sizeof(float) * n_td);
  float* attn = malloc(sizeof(float) * n_tt);
  CHECK_OK(emunpu_device_download(dev, d_y, y, sizeof(float) * n_td));
  CHECK_OK(emunpu_device_download(dev, d_s, attn, sizeof(float) * n_tt));

  // --- Host reference ---------------------------------------------------
  double* r_x = malloc(sizeof(double) * n_td);
  double* r_xn = malloc(sizeof(double) * n_td);
  double* r_q = malloc(sizeof(double) * n_td);
  double* r_k = malloc(sizeof(double) * n_td);
  double* r_v = malloc(sizeof(double) * n_td);
  double* r_s = malloc(sizeof(double) * n_tt);
  double* r_ctx = malloc(sizeof(double) * n_td);
  double* r_h = malloc(sizeof(double) * n_td);
  double* r_hn = malloc(sizeof(double) * n_td);
  double* r_ff = malloc(sizeof(double) * n_tff);
  double* r_y = malloc(sizeof(double) * n_td);

  for (uint32_t t = 0; t < T; ++t) {
    for (uint32_t d = 0; d < D; ++d) {
      r_x[(size_t)t * D + d] = emb[(size_t)tokens[t] * D + d];
    }
  }
  ref_layernorm(r_x, g1, be1, r_xn, T, D, eps);
  ref_linear(r_xn, wq, NULL, r_q, T, D, D);
  ref_linear(r_xn, wk, NULL, r_k, T, D, D);
  ref_linear(r_xn, wv, NULL, r_v, T, D, D);
  for (uint32_t i = 0; i < T; ++i) {
    for (uint32_t j = 0; j < T; ++j) {
      double acc = 0.0;
      for (uint32_t d = 0; d < D; ++d) {
        acc += r_q[(size_t)i * D + d] * r_k[(size_t)j * D + d];
      }
      r_s[(size_t)i * T + j] = acc * (double)inv_sqrt_d;
    }
  }
  ref_softmax_rows(r_s, T, T);
  for (uint32_t i = 0; i < T; ++i) {
    for (uint32_t d = 0; d < D; ++d) {
      double acc = 0.0;
      for (uint32_t j = 0; j < T; ++j) {
        acc += r_s[(size_t)i * T + j] * r_v[(size_t)j * D + d];
      }
      r_ctx[(size_t)i * D + d] = acc;
    }
  }
  ref_linear(r_ctx, wo, bo, r_h, T, D, D);
  for (uint32_t i = 0; i < n_td; ++i) r_h[i] += r_x[i];
  ref_layernorm(r_h, g2, be2, r_hn, T, D, eps);
  ref_linear(r_hn, w1, b1, r_ff, T, FF, D);
  for (uint32_t i = 0; i < n_tff; ++i) r_ff[i] = ref_gelu(r_ff[i]);
  ref_linear(r_ff, w2, b2, r_y, T, D, FF);
  for (uint32_t i = 0; i < n_td; ++i) r_y[i] += r_h[i];

  // --- Report -----------------------------------------------------------
  double max_abs_err = 0.0, max_attn_err = 0.0, worst_row_sum = 0.0;
  for (uint32_t i = 0; i < n_td; ++i) {
    double e = fabs((double)y[i] - r_y[i]);
    if (e > max_abs_err) max_abs_err = e;
  }
  for (uint32_t i = 0; i < n_tt; ++i) {
    double e = fabs((double)attn[i] - r_s[i]);
    if (e > max_attn_err) max_attn_err = e;
  }
  for (uint32_t i = 0; i < T; ++i) {
    double s = 0.0;
    for (uint32_t j = 0; j < T; ++j) s += attn[(size_t)i * T + j];
    if (fabs(s - 1.0) > worst_row_sum) worst_row_sum = fabs(s - 1.0);
  }

  printf("attention row 0 (first 8 of %d):", T);
  for (uint32_t j = 0; j < 8; ++j) printf(" %.4f", attn[j]);
  printf("\noutput y[0, 0:6] =");
  for (uint32_t d = 0; d < 6; ++d) printf(" % .4f", y[d]);
  printf(
      "\n\nattention weights: max abs error %.3e, worst row-sum drift %.3e\n",
      max_attn_err, worst_row_sum);
  printf("block output:      max abs error %.3e\n", max_abs_err);
  int pass = max_abs_err < 1e-4 && max_attn_err < 1e-5 && worst_row_sum < 1e-5;
  printf("-> %s\n", pass ? "PASS" : "FAIL");

  free(emb);
  free(wq);
  free(wk);
  free(wv);
  free(wo);
  free(bo);
  free(w1);
  free(b1);
  free(w2);
  free(b2);
  free(y);
  free(attn);
  free(r_x);
  free(r_xn);
  free(r_q);
  free(r_k);
  free(r_v);
  free(r_s);
  free(r_ctx);
  free(r_h);
  free(r_hn);
  free(r_ff);
  free(r_y);
  emunpu_device_destroy(dev);
  return pass ? 0 : 1;
}
