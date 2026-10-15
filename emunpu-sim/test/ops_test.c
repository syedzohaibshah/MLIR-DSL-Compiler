// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Functional test for the ML operation set: activations, extended reductions,
// softmax, layer/RMS normalization, transpose, im2col, conv2d, pooling, gather,
// and linear layers with a fused activation epilogue.
//
// Every expectation is an *independently written* host reference (plain loops
// over host arrays with the textbook formula) -- the test never calls the
// simulator's own helpers to compute what it is checking, so a wrong kernel
// cannot agree with a wrong expectation. Problem sizes are deliberately not
// multiples of the core count so the multi-core range/row splitting is
// exercised with ragged edges.
//
//   make test          # or: ctest --test-dir build -R ops_test
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emunpu/device.h"

static int g_failures = 0;
static int g_checks = 0;
static emunpu_device_t* g_dev = NULL;

#define CHECK(cond, ...)                                   \
  do {                                                     \
    ++g_checks;                                            \
    if (!(cond)) {                                         \
      ++g_failures;                                        \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

#define CHECK_OK(expr)                                           \
  do {                                                           \
    emunpu_status_t _s = (expr);                                 \
    CHECK(_s == EMUNPU_OK, "expected OK from %s, got %s", #expr, \
          emunpu_status_name(_s));                               \
  } while (0)

#define CHECK_ERR(expr)                                                 \
  do {                                                                  \
    emunpu_status_t _s = (expr);                                        \
    CHECK(_s != EMUNPU_OK, "expected an error from %s, got OK", #expr); \
  } while (0)

//===----------------------------------------------------------------------===//
// Host-side helpers
//===----------------------------------------------------------------------===//

// Deterministic values in roughly [-1.5, 1.5).
static void fill_rand(float* p, size_t n, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    p[i] = (float)((int)((s >> 16) & 0x3ff) - 512) * (1.0f / 341.0f);
  }
}

static emunpu_tensor_id_t alloc_n(uint32_t n, emunpu_dtype_t dtype,
                                  const char* name) {
  emunpu_tensor_id_t id = 0;
  CHECK_OK(emunpu_device_alloc(g_dev, dtype, 1, &n, name, &id));
  return id;
}

static emunpu_tensor_id_t upload_f32(const float* data, uint32_t n,
                                     const char* name) {
  emunpu_tensor_id_t id = alloc_n(n, EMUNPU_DT_FP32, name);
  CHECK_OK(emunpu_device_upload(g_dev, id, data, (uint64_t)n * sizeof(float)));
  return id;
}

static void download_f32(emunpu_tensor_id_t id, float* dst, uint32_t n) {
  CHECK_OK(emunpu_device_download(g_dev, id, dst, (uint64_t)n * sizeof(float)));
}

// Mixed absolute/relative comparison: |got - want| <= tol * (1 + |want|).
static int values_close(double got, double want, double tol) {
  if (isnan(got) || isnan(want)) return isnan(got) && isnan(want);
  return fabs(got - want) <= tol * (1.0 + fabs(want));
}

static void check_array(const char* label, const float* got, const double* want,
                        uint32_t n, double tol) {
  uint32_t bad = 0;
  uint32_t first = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!values_close((double)got[i], want[i], tol)) {
      if (!bad) first = i;
      ++bad;
    }
  }
  CHECK(bad == 0, "%s: %u/%u elements differ (first at %u: got %g want %g)",
        label, bad, n, first, bad ? (double)got[first] : 0.0,
        bad ? want[first] : 0.0);
}

//===----------------------------------------------------------------------===//
// Activations
//===----------------------------------------------------------------------===//

// The reference transfer functions, written out longhand from the definitions
// (PyTorch semantics) rather than shared with the device implementation.
static double ref_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double ref_softplus(double x) { return log(1.0 + exp(x)); }

static double ref_act(emunpu_act_kind_t kind, double x, double alpha,
                      double beta) {
  switch (kind) {
    case EMUNPU_ACT_RELU:
      return x < 0.0 ? 0.0 : x;
    case EMUNPU_ACT_LEAKY_RELU:
      return x < 0.0 ? alpha * x : x;
    case EMUNPU_ACT_RELU6:
      return x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
    case EMUNPU_ACT_CLAMP:
      return x < alpha ? alpha : (x > beta ? beta : x);
    case EMUNPU_ACT_SIGMOID:
      return ref_sigmoid(x);
    case EMUNPU_ACT_TANH:
      return (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    case EMUNPU_ACT_GELU:
      return x * 0.5 * (1.0 + erf(x / sqrt(2.0)));
    case EMUNPU_ACT_GELU_TANH:
      return 0.5 * x *
             (1.0 + tanh(sqrt(2.0 / acos(-1.0)) * (x + 0.044715 * x * x * x)));
    case EMUNPU_ACT_SILU:
      return x * ref_sigmoid(x);
    case EMUNPU_ACT_ELU:
      return x < 0.0 ? alpha * (exp(x) - 1.0) : x;
    case EMUNPU_ACT_SOFTPLUS:
      return ref_softplus(x);
    case EMUNPU_ACT_HARDSIGMOID: {
      double v = x / 6.0 + 0.5;
      return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    case EMUNPU_ACT_HARDSWISH: {
      double v = x / 6.0 + 0.5;
      return x * (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
    }
    case EMUNPU_ACT_MISH:
      return x * tanh(ref_softplus(x));
    case EMUNPU_ACT_EXP:
      return exp(x);
    case EMUNPU_ACT_LOG:
      return log(x);
    case EMUNPU_ACT_SQRT:
      return sqrt(x);
    case EMUNPU_ACT_RSQRT:
      return 1.0 / sqrt(x);
    case EMUNPU_ACT_ABS:
      return x < 0.0 ? -x : x;
    case EMUNPU_ACT_NEG:
      return 0.0 - x;
    case EMUNPU_ACT_RECIP:
      return 1.0 / x;
    case EMUNPU_ACT_SIGN:
      return x < 0.0 ? -1.0 : (x > 0.0 ? 1.0 : 0.0);
    case EMUNPU_ACT_SQUARE:
      return x * x;
    case EMUNPU_ACT_ERF:
      return erf(x);
    default:
      return x;
  }
}

// Kinds whose domain excludes zero/negatives get a strictly positive input set.
static int act_needs_positive(emunpu_act_kind_t kind) {
  return kind == EMUNPU_ACT_LOG || kind == EMUNPU_ACT_SQRT ||
         kind == EMUNPU_ACT_RSQRT || kind == EMUNPU_ACT_RECIP;
}

static void test_activations(void) {
  // 137 elements: not a multiple of 10 cores, so the last core gets a short
  // range and the split is exercised with a ragged tail.
  enum { N = 137 };
  float mixed[N], positive[N], got[N];
  double want[N];
  for (uint32_t i = 0; i < N; ++i) {
    mixed[i] = (float)((double)i * 0.17 - 11.0);  // spans -11 .. +12, incl. 0
    positive[i] = (float)(0.05 + (double)i * 0.13);
  }
  mixed[7] = 0.0f;  // pin an exact zero (sign/relu boundary)

  emunpu_tensor_id_t a_mixed = upload_f32(mixed, N, "act.mixed");
  emunpu_tensor_id_t a_pos = upload_f32(positive, N, "act.pos");
  emunpu_tensor_id_t c = alloc_n(N, EMUNPU_DT_FP32, "act.out");

  for (int k = 0; k < EMUNPU_ACT_COUNT; ++k) {
    emunpu_act_kind_t kind = (emunpu_act_kind_t)k;
    int positive_only = act_needs_positive(kind);
    const float* src = positive_only ? positive : mixed;
    // alpha/beta only matter for LEAKY_RELU / ELU / CLAMP; pass values that are
    // distinguishable from the defaults so a dropped parameter shows up.
    double alpha = (kind == EMUNPU_ACT_CLAMP) ? -1.25 : 0.125;
    double beta = 2.5;
    CHECK_OK(emunpu_activation(g_dev, NULL, c, positive_only ? a_pos : a_mixed,
                               N, kind, alpha, beta));
    download_f32(c, got, N);
    for (uint32_t i = 0; i < N; ++i) {
      want[i] = ref_act(kind, (double)src[i], alpha, beta);
    }
    char label[64];
    snprintf(label, sizeof(label), "activation %s", emunpu_act_name(kind));
    check_array(label, got, want, N, 1e-6);
  }

  // The activation unit is dtype-generic: fp16 in, fp16 out (wider tolerance
  // because fp16 has ~3 decimal digits).
  {
    emunpu_tensor_id_t h_in = alloc_n(N, EMUNPU_DT_FP16, "act.fp16.in");
    emunpu_tensor_id_t h_out = alloc_n(N, EMUNPU_DT_FP16, "act.fp16.out");
    uint16_t half_in[N], half_out[N];
    for (uint32_t i = 0; i < N; ++i) half_in[i] = emunpu_f32_to_fp16(mixed[i]);
    CHECK_OK(emunpu_device_upload(g_dev, h_in, half_in, sizeof(half_in)));
    CHECK_OK(emunpu_activation(g_dev, NULL, h_out, h_in, N, EMUNPU_ACT_GELU,
                               0.0, 0.0));
    CHECK_OK(emunpu_device_download(g_dev, h_out, half_out, sizeof(half_out)));
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      double w = ref_act(EMUNPU_ACT_GELU,
                         (double)emunpu_fp16_to_f32(half_in[i]), 0.0, 0.0);
      if (!values_close((double)emunpu_fp16_to_f32(half_out[i]), w, 1e-3))
        ++bad;
    }
    CHECK(bad == 0, "fp16 gelu: %u/%u elements differ", bad, (uint32_t)N);
    emunpu_device_free(g_dev, h_in);
    emunpu_device_free(g_dev, h_out);
  }

  // An unknown activation kind must be rejected, not silently treated as copy.
  CHECK_ERR(emunpu_activation(g_dev, NULL, c, a_mixed, N,
                              (emunpu_act_kind_t)EMUNPU_ACT_COUNT, 0.0, 0.0));

  emunpu_device_free(g_dev, a_mixed);
  emunpu_device_free(g_dev, a_pos);
  emunpu_device_free(g_dev, c);
}

//===----------------------------------------------------------------------===//
// Extended reductions
//===----------------------------------------------------------------------===//

static void test_reductions(void) {
  enum { ROWS = 23, WIDTH = 9 };  // 23 rows over 10 cores -> ragged split
  float in[ROWS * WIDTH], got[ROWS];
  double want[ROWS];
  fill_rand(in, ROWS * WIDTH, 11);
  // Keep the product in a sane range so PROD stays comparable in fp32.
  for (uint32_t i = 0; i < ROWS * WIDTH; ++i) in[i] = 0.5f + 0.5f * in[i];

  emunpu_tensor_id_t a = upload_f32(in, ROWS * WIDTH, "red.in");
  emunpu_tensor_id_t c = alloc_n(ROWS, EMUNPU_DT_FP32, "red.out");
  emunpu_tensor_id_t ci = alloc_n(ROWS, EMUNPU_DT_INT32, "red.idx");

  const emunpu_reduce_kind_t value_kinds[] = {
      EMUNPU_REDUCE_SUM, EMUNPU_REDUCE_MAX, EMUNPU_REDUCE_MIN,
      EMUNPU_REDUCE_MEAN, EMUNPU_REDUCE_PROD};
  for (size_t k = 0; k < sizeof(value_kinds) / sizeof(value_kinds[0]); ++k) {
    emunpu_reduce_kind_t kind = value_kinds[k];
    for (uint32_t r = 0; r < ROWS; ++r) {
      const float* row = in + (size_t)r * WIDTH;
      double acc = row[0];
      for (uint32_t j = 1; j < WIDTH; ++j) {
        double v = row[j];
        if (kind == EMUNPU_REDUCE_SUM || kind == EMUNPU_REDUCE_MEAN) {
          acc += v;
        } else if (kind == EMUNPU_REDUCE_PROD) {
          acc *= v;
        } else if (kind == EMUNPU_REDUCE_MAX) {
          if (v > acc) acc = v;
        } else if (v < acc) {
          acc = v;
        }
      }
      want[r] = (kind == EMUNPU_REDUCE_MEAN) ? acc / (double)WIDTH : acc;
    }
    CHECK_OK(emunpu_reduce(g_dev, NULL, c, a, ROWS, WIDTH, kind));
    download_f32(c, got, ROWS);
    char label[48];
    snprintf(label, sizeof(label), "reduce %s", emunpu_reduce_name(kind));
    check_array(label, got, want, ROWS, 1e-6);
  }

  // ARGMAX / ARGMIN emit indices into an integer tensor.
  for (int pass = 0; pass < 2; ++pass) {
    emunpu_reduce_kind_t kind =
        pass ? EMUNPU_REDUCE_ARGMIN : EMUNPU_REDUCE_ARGMAX;
    int32_t idx[ROWS];
    CHECK_OK(emunpu_reduce(g_dev, NULL, ci, a, ROWS, WIDTH, kind));
    CHECK_OK(emunpu_device_download(g_dev, ci, idx, sizeof(idx)));
    uint32_t bad = 0;
    for (uint32_t r = 0; r < ROWS; ++r) {
      const float* row = in + (size_t)r * WIDTH;
      uint32_t best = 0;
      for (uint32_t j = 1; j < WIDTH; ++j) {
        if (pass ? (row[j] < row[best]) : (row[j] > row[best])) best = j;
      }
      if (idx[r] != (int32_t)best) ++bad;
    }
    CHECK(bad == 0, "reduce %s: %u/%u rows differ", emunpu_reduce_name(kind),
          bad, (uint32_t)ROWS);
  }

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, c);
  emunpu_device_free(g_dev, ci);
}

//===----------------------------------------------------------------------===//
// Elementwise pow (the new binary kind)
//===----------------------------------------------------------------------===//

static void test_ew_pow(void) {
  enum { N = 61 };
  float base[N], expo[N], got[N];
  double want[N];
  for (uint32_t i = 0; i < N; ++i) {
    base[i] = 0.25f + (float)i * 0.1f;
    expo[i] = (float)((int)(i % 5) - 2);  // -2 .. 2
    want[i] = pow((double)base[i], (double)expo[i]);
  }
  emunpu_tensor_id_t a = upload_f32(base, N, "pow.a");
  emunpu_tensor_id_t b = upload_f32(expo, N, "pow.b");
  emunpu_tensor_id_t c = alloc_n(N, EMUNPU_DT_FP32, "pow.c");
  CHECK_OK(emunpu_vector_op(g_dev, NULL, c, a, b, N, EMUNPU_EW_POW, 0));
  download_f32(c, got, N);
  check_array("elementwise pow", got, want, N, 1e-6);
  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, b);
  emunpu_device_free(g_dev, c);
}

//===----------------------------------------------------------------------===//
// Softmax / log-softmax
//===----------------------------------------------------------------------===//

static void test_softmax(void) {
  enum { ROWS = 17, WIDTH = 13 };  // ragged over 10 cores
  float in[ROWS * WIDTH], got[ROWS * WIDTH];
  double want[ROWS * WIDTH];
  fill_rand(in, ROWS * WIDTH, 5);
  // A row with a large offset proves the max-subtraction keeps exp() in range.
  for (uint32_t j = 0; j < WIDTH; ++j) in[3 * WIDTH + j] += 400.0f;

  emunpu_tensor_id_t a = upload_f32(in, ROWS * WIDTH, "sm.in");
  emunpu_tensor_id_t c = alloc_n(ROWS * WIDTH, EMUNPU_DT_FP32, "sm.out");

  for (int log_variant = 0; log_variant < 2; ++log_variant) {
    for (uint32_t r = 0; r < ROWS; ++r) {
      const float* row = in + (size_t)r * WIDTH;
      double max = row[0];
      for (uint32_t j = 1; j < WIDTH; ++j)
        if (row[j] > max) max = row[j];
      double sum = 0.0;
      for (uint32_t j = 0; j < WIDTH; ++j) sum += exp((double)row[j] - max);
      for (uint32_t j = 0; j < WIDTH; ++j) {
        double shifted = (double)row[j] - max;
        want[r * WIDTH + j] =
            log_variant ? shifted - log(sum) : exp(shifted) / sum;
      }
    }
    CHECK_OK(emunpu_softmax(g_dev, NULL, c, a, ROWS, WIDTH, log_variant));
    download_f32(c, got, ROWS * WIDTH);
    check_array(log_variant ? "log_softmax" : "softmax", got, want,
                ROWS * WIDTH, 1e-6);
    if (!log_variant) {
      // Independent property: every row must sum to 1.
      double worst = 0.0;
      for (uint32_t r = 0; r < ROWS; ++r) {
        double s = 0.0;
        for (uint32_t j = 0; j < WIDTH; ++j) s += got[r * WIDTH + j];
        if (fabs(s - 1.0) > worst) worst = fabs(s - 1.0);
      }
      CHECK(worst < 1e-5, "softmax rows do not sum to 1 (worst %g)", worst);
    }
  }

  CHECK_ERR(emunpu_softmax(g_dev, NULL, c, a, ROWS, 0, 0));

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, c);
}

//===----------------------------------------------------------------------===//
// Layer / RMS normalization
//===----------------------------------------------------------------------===//

static void test_norm(void) {
  enum { ROWS = 14, WIDTH = 11 };
  const double eps = 1e-5;
  float in[ROWS * WIDTH], gamma[WIDTH], beta[WIDTH], got[ROWS * WIDTH];
  double want[ROWS * WIDTH];
  fill_rand(in, ROWS * WIDTH, 7);
  for (uint32_t j = 0; j < WIDTH; ++j) {
    gamma[j] = 0.5f + 0.1f * (float)j;
    beta[j] = -0.25f + 0.05f * (float)j;
  }

  emunpu_tensor_id_t a = upload_f32(in, ROWS * WIDTH, "ln.in");
  emunpu_tensor_id_t g = upload_f32(gamma, WIDTH, "ln.gamma");
  emunpu_tensor_id_t b = upload_f32(beta, WIDTH, "ln.beta");
  emunpu_tensor_id_t c = alloc_n(ROWS * WIDTH, EMUNPU_DT_FP32, "ln.out");

  // --- layernorm with the affine terms.
  for (uint32_t r = 0; r < ROWS; ++r) {
    const float* row = in + (size_t)r * WIDTH;
    double mean = 0.0;
    for (uint32_t j = 0; j < WIDTH; ++j) mean += row[j];
    mean /= (double)WIDTH;
    double var = 0.0;
    for (uint32_t j = 0; j < WIDTH; ++j)
      var += (row[j] - mean) * (row[j] - mean);
    var /= (double)WIDTH;
    for (uint32_t j = 0; j < WIDTH; ++j) {
      want[r * WIDTH + j] =
          (row[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j];
    }
  }
  CHECK_OK(emunpu_layernorm(g_dev, NULL, c, a, g, b, ROWS, WIDTH, eps));
  download_f32(c, got, ROWS * WIDTH);
  check_array("layernorm (affine)", got, want, ROWS * WIDTH, 1e-5);

  // --- layernorm without gamma/beta: each row must have mean 0, variance 1.
  CHECK_OK(emunpu_layernorm(g_dev, NULL, c, a, 0, 0, ROWS, WIDTH, eps));
  download_f32(c, got, ROWS * WIDTH);
  double worst_mean = 0.0, worst_var = 0.0;
  for (uint32_t r = 0; r < ROWS; ++r) {
    double m = 0.0, v = 0.0;
    for (uint32_t j = 0; j < WIDTH; ++j) m += got[r * WIDTH + j];
    m /= (double)WIDTH;
    for (uint32_t j = 0; j < WIDTH; ++j) {
      double d = got[r * WIDTH + j] - m;
      v += d * d;
    }
    v /= (double)WIDTH;
    if (fabs(m) > worst_mean) worst_mean = fabs(m);
    if (fabs(v - 1.0) > worst_var) worst_var = fabs(v - 1.0);
  }
  CHECK(worst_mean < 1e-5, "layernorm row mean not 0 (worst %g)", worst_mean);
  CHECK(worst_var < 1e-4, "layernorm row variance not 1 (worst %g)", worst_var);

  // --- rmsnorm with gamma (no centering).
  for (uint32_t r = 0; r < ROWS; ++r) {
    const float* row = in + (size_t)r * WIDTH;
    double ms = 0.0;
    for (uint32_t j = 0; j < WIDTH; ++j) ms += (double)row[j] * row[j];
    ms /= (double)WIDTH;
    for (uint32_t j = 0; j < WIDTH; ++j) {
      want[r * WIDTH + j] = row[j] / sqrt(ms + eps) * gamma[j];
    }
  }
  CHECK_OK(emunpu_rmsnorm(g_dev, NULL, c, a, g, ROWS, WIDTH, eps));
  download_f32(c, got, ROWS * WIDTH);
  check_array("rmsnorm (gamma)", got, want, ROWS * WIDTH, 1e-5);

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, g);
  emunpu_device_free(g_dev, b);
  emunpu_device_free(g_dev, c);
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

static void test_transpose(void) {
  enum { ROWS = 37, COLS = 19 };  // non-square, neither dim a multiple of 10
  float in[ROWS * COLS], got[ROWS * COLS];
  double want[ROWS * COLS];
  fill_rand(in, ROWS * COLS, 3);
  for (uint32_t i = 0; i < ROWS; ++i) {
    for (uint32_t j = 0; j < COLS; ++j) want[j * ROWS + i] = in[i * COLS + j];
  }
  emunpu_tensor_id_t a = upload_f32(in, ROWS * COLS, "tr.in");
  emunpu_tensor_id_t c = alloc_n(ROWS * COLS, EMUNPU_DT_FP32, "tr.out");
  CHECK_OK(emunpu_transpose(g_dev, NULL, c, a, ROWS, COLS));
  download_f32(c, got, ROWS * COLS);
  check_array("transpose", got, want, ROWS * COLS, 0.0);

  // Transposing back must reproduce the input exactly (it is a pure
  // permutation).
  emunpu_tensor_id_t back = alloc_n(ROWS * COLS, EMUNPU_DT_FP32, "tr.back");
  CHECK_OK(emunpu_transpose(g_dev, NULL, back, c, COLS, ROWS));
  download_f32(back, got, ROWS * COLS);
  int exact = 1;
  for (uint32_t i = 0; i < ROWS * COLS; ++i)
    if (got[i] != in[i]) exact = 0;
  CHECK(exact, "double transpose is not the identity");

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, c);
  emunpu_device_free(g_dev, back);
}

//===----------------------------------------------------------------------===//
// im2col
//===----------------------------------------------------------------------===//

static void test_im2col(void) {
  emunpu_window_t w = {0};
  w.batch = 2;
  w.channels = 3;
  w.in_h = 5;
  w.in_w = 7;
  w.kernel_h = 3;
  w.kernel_w = 2;
  w.stride_h = 2;
  w.stride_w = 1;
  w.pad_h = 1;
  w.pad_w = 1;
  uint32_t oh = 0, ow = 0;
  emunpu_window_output(&w, &oh, &ow);
  // (5 + 2*1 - 3)/2 + 1 = 3 rows, (7 + 2*1 - 2)/1 + 1 = 8 columns.
  CHECK(oh == 3 && ow == 8, "unexpected im2col output extents %ux%u", oh, ow);

  uint32_t image = w.channels * w.in_h * w.in_w;
  uint32_t patches = oh * ow;
  uint32_t width = w.channels * w.kernel_h * w.kernel_w;
  uint32_t rows = w.batch * patches;

  float* in = malloc(sizeof(float) * image * w.batch);
  float* got = malloc(sizeof(float) * (size_t)rows * width);
  double* want = malloc(sizeof(double) * (size_t)rows * width);
  fill_rand(in, (size_t)image * w.batch, 13);

  for (uint32_t bi = 0; bi < w.batch; ++bi) {
    for (uint32_t p = 0; p < patches; ++p) {
      uint32_t oy = p / ow, ox = p % ow;
      uint32_t col = 0;
      for (uint32_t ch = 0; ch < w.channels; ++ch) {
        for (uint32_t ky = 0; ky < w.kernel_h; ++ky) {
          for (uint32_t kx = 0; kx < w.kernel_w; ++kx, ++col) {
            long iy = (long)oy * w.stride_h - w.pad_h + ky;
            long ix = (long)ox * w.stride_w - w.pad_w + kx;
            double v = 0.0;
            if (iy >= 0 && iy < (long)w.in_h && ix >= 0 && ix < (long)w.in_w) {
              v = in[bi * image + (ch * w.in_h + iy) * w.in_w + ix];
            }
            want[((size_t)bi * patches + p) * width + col] = v;
          }
        }
      }
    }
  }

  emunpu_tensor_id_t a = upload_f32(in, image * w.batch, "i2c.in");
  emunpu_tensor_id_t c = alloc_n(rows * width, EMUNPU_DT_FP32, "i2c.out");
  CHECK_OK(emunpu_im2col(g_dev, NULL, c, a, &w));
  download_f32(c, got, rows * width);
  check_array("im2col", got, want, rows * width, 0.0);

  // A window larger than the padded input is not representable.
  emunpu_window_t bad = w;
  bad.kernel_h = 99;
  CHECK_ERR(emunpu_im2col(g_dev, NULL, c, a, &bad));

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, c);
  free(in);
  free(got);
  free(want);
}

//===----------------------------------------------------------------------===//
// conv2d
//===----------------------------------------------------------------------===//

// Direct convolution reference: NCHW in, [out_channels, C, kh, kw] weight.
static void ref_conv2d(const float* in, const float* weight, const float* bias,
                       double* out, const emunpu_window_t* w,
                       uint32_t out_channels, uint32_t oh, uint32_t ow,
                       emunpu_act_kind_t act, int has_act) {
  uint32_t ksize = w->channels * w->kernel_h * w->kernel_w;
  for (uint32_t bi = 0; bi < w->batch; ++bi) {
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
      for (uint32_t oy = 0; oy < oh; ++oy) {
        for (uint32_t ox = 0; ox < ow; ++ox) {
          double acc = bias ? bias[oc] : 0.0;
          for (uint32_t ch = 0; ch < w->channels; ++ch) {
            for (uint32_t ky = 0; ky < w->kernel_h; ++ky) {
              long iy = (long)oy * w->stride_h - w->pad_h + ky;
              if (iy < 0 || iy >= (long)w->in_h) continue;
              for (uint32_t kx = 0; kx < w->kernel_w; ++kx) {
                long ix = (long)ox * w->stride_w - w->pad_w + kx;
                if (ix < 0 || ix >= (long)w->in_w) continue;
                double x = in[(bi * w->channels + ch) * w->in_h * w->in_w +
                              iy * w->in_w + ix];
                double k = weight[oc * ksize +
                                  (ch * w->kernel_h + ky) * w->kernel_w + kx];
                acc += x * k;
              }
            }
          }
          if (has_act) acc = ref_act(act, acc, 0.125, 2.5);
          out[((bi * out_channels + oc) * oh + oy) * ow + ox] = acc;
        }
      }
    }
  }
}

static void run_conv_case(const char* label, emunpu_window_t w,
                          uint32_t out_channels, int with_bias, int with_act,
                          emunpu_act_kind_t act_kind) {
  uint32_t oh = 0, ow = 0;
  emunpu_window_output(&w, &oh, &ow);
  CHECK(oh && ow, "%s: window does not fit", label);
  if (!oh || !ow) return;

  size_t in_elems = (size_t)w.batch * w.channels * w.in_h * w.in_w;
  size_t k_elems = (size_t)out_channels * w.channels * w.kernel_h * w.kernel_w;
  size_t out_elems = (size_t)w.batch * out_channels * oh * ow;

  float* in = malloc(sizeof(float) * in_elems);
  float* weight = malloc(sizeof(float) * k_elems);
  float* bias = malloc(sizeof(float) * out_channels);
  float* got = malloc(sizeof(float) * out_elems);
  double* want = malloc(sizeof(double) * out_elems);
  fill_rand(in, in_elems, 21);
  fill_rand(weight, k_elems, 22);
  fill_rand(bias, out_channels, 23);

  emunpu_act_t act = {act_kind, 0.125, 2.5};
  ref_conv2d(in, weight, with_bias ? bias : NULL, want, &w, out_channels, oh,
             ow, act_kind, with_act);

  emunpu_tensor_id_t d_in = upload_f32(in, (uint32_t)in_elems, "conv.in");
  emunpu_tensor_id_t d_w = upload_f32(weight, (uint32_t)k_elems, "conv.w");
  emunpu_tensor_id_t d_b = upload_f32(bias, out_channels, "conv.b");
  emunpu_tensor_id_t d_out =
      alloc_n((uint32_t)out_elems, EMUNPU_DT_FP32, "conv.out");

  CHECK_OK(emunpu_conv2d(g_dev, NULL, d_out, d_in, d_w, with_bias ? d_b : 0, &w,
                         out_channels, with_act ? &act : NULL));
  download_f32(d_out, got, (uint32_t)out_elems);
  check_array(label, got, want, (uint32_t)out_elems, 1e-5);

  emunpu_device_free(g_dev, d_in);
  emunpu_device_free(g_dev, d_w);
  emunpu_device_free(g_dev, d_b);
  emunpu_device_free(g_dev, d_out);
  free(in);
  free(weight);
  free(bias);
  free(got);
  free(want);
}

static void test_conv2d(void) {
  // 3x3 same-padding convolution with bias + ReLU, batch 2 -- the shape a CNN's
  // first layer actually has. K = 3*3*3 = 27 > MMA_DIM, so the GEMM accumulates
  // over multiple K-tiles across the cores.
  emunpu_window_t w = {0};
  w.batch = 2;
  w.channels = 3;
  w.in_h = 9;
  w.in_w = 11;
  w.kernel_h = 3;
  w.kernel_w = 3;
  w.stride_h = 1;
  w.stride_w = 1;
  w.pad_h = 1;
  w.pad_w = 1;
  run_conv_case("conv2d 3x3 pad1 +bias +relu", w, 6, 1, 1, EMUNPU_ACT_RELU);

  // Strided, unpadded, non-square kernel, no bias, no activation.
  emunpu_window_t s = {0};
  s.batch = 1;
  s.channels = 2;
  s.in_h = 12;
  s.in_w = 10;
  s.kernel_h = 3;
  s.kernel_w = 2;
  s.stride_h = 2;
  s.stride_w = 3;
  s.pad_h = 0;
  s.pad_w = 0;
  run_conv_case("conv2d 3x2 stride2x3 bare", s, 4, 0, 0, EMUNPU_ACT_RELU);

  // 1x1 (pointwise) convolution with a GELU epilogue.
  emunpu_window_t p = {0};
  p.batch = 2;
  p.channels = 5;
  p.in_h = 4;
  p.in_w = 6;
  p.kernel_h = 1;
  p.kernel_w = 1;
  p.stride_h = 1;
  p.stride_w = 1;
  run_conv_case("conv2d 1x1 +gelu", p, 3, 1, 1, EMUNPU_ACT_GELU);

  // The scratch buffers a conv allocates must be released again: allocate,
  // convolve repeatedly, and confirm global memory use returns to the baseline.
  {
    emunpu_global_mem_t* gm = emunpu_device_global_mem(g_dev);
    uint64_t before = gm->bytes_in_use;
    run_conv_case("conv2d (leak check)", w, 6, 1, 1, EMUNPU_ACT_RELU);
    CHECK(gm->bytes_in_use == before,
          "conv2d leaked global memory (%llu -> %llu bytes)",
          (unsigned long long)before, (unsigned long long)gm->bytes_in_use);
  }
}

//===----------------------------------------------------------------------===//
// Pooling
//===----------------------------------------------------------------------===//

static void test_pool2d(void) {
  emunpu_window_t w = {0};
  w.batch = 3;
  w.channels = 4;  // 12 planes over 10 cores -> ragged split
  w.in_h = 7;
  w.in_w = 5;
  w.kernel_h = 3;
  w.kernel_w = 3;
  w.stride_h = 2;
  w.stride_w = 2;
  w.pad_h = 1;
  w.pad_w = 1;
  uint32_t oh = 0, ow = 0;
  emunpu_window_output(&w, &oh, &ow);
  CHECK(oh == 4 && ow == 3, "unexpected pool output extents %ux%u", oh, ow);

  uint32_t planes = w.batch * w.channels;
  size_t in_elems = (size_t)planes * w.in_h * w.in_w;
  size_t out_elems = (size_t)planes * oh * ow;
  float* in = malloc(sizeof(float) * in_elems);
  float* got = malloc(sizeof(float) * out_elems);
  double* want = malloc(sizeof(double) * out_elems);
  fill_rand(in, in_elems, 31);

  emunpu_tensor_id_t a = upload_f32(in, (uint32_t)in_elems, "pool.in");
  emunpu_tensor_id_t c =
      alloc_n((uint32_t)out_elems, EMUNPU_DT_FP32, "pool.out");

  for (int is_max = 1; is_max >= 0; --is_max) {
    for (uint32_t p = 0; p < planes; ++p) {
      for (uint32_t oy = 0; oy < oh; ++oy) {
        for (uint32_t ox = 0; ox < ow; ++ox) {
          double best = 0.0, sum = 0.0;
          uint32_t taps = 0;
          for (uint32_t ky = 0; ky < w.kernel_h; ++ky) {
            long iy = (long)oy * w.stride_h - w.pad_h + ky;
            if (iy < 0 || iy >= (long)w.in_h) continue;
            for (uint32_t kx = 0; kx < w.kernel_w; ++kx) {
              long ix = (long)ox * w.stride_w - w.pad_w + kx;
              if (ix < 0 || ix >= (long)w.in_w) continue;
              double v = in[p * w.in_h * w.in_w + iy * w.in_w + ix];
              if (taps == 0 || v > best) best = v;
              sum += v;
              ++taps;
            }
          }
          want[(p * oh + oy) * ow + ox] =
              is_max ? best : (taps ? sum / taps : 0.0);
        }
      }
    }
    CHECK_OK(emunpu_pool2d(g_dev, NULL, c, a, &w,
                           is_max ? EMUNPU_POOL_MAX : EMUNPU_POOL_AVG));
    download_f32(c, got, (uint32_t)out_elems);
    check_array(is_max ? "maxpool2d" : "avgpool2d", got, want,
                (uint32_t)out_elems, 1e-6);
  }

  CHECK_ERR(emunpu_pool2d(g_dev, NULL, c, a, &w,
                          (emunpu_pool_kind_t)EMUNPU_POOL_COUNT));

  emunpu_device_free(g_dev, a);
  emunpu_device_free(g_dev, c);
  free(in);
  free(got);
  free(want);
}

//===----------------------------------------------------------------------===//
// Gather (embedding lookup)
//===----------------------------------------------------------------------===//

static void test_gather(void) {
  enum { TABLE_ROWS = 16, WIDTH = 8, COUNT = 23 };
  float table[TABLE_ROWS * WIDTH], got[COUNT * WIDTH];
  double want[COUNT * WIDTH];
  int32_t indices[COUNT];
  fill_rand(table, TABLE_ROWS * WIDTH, 41);
  for (uint32_t i = 0; i < COUNT; ++i) {
    indices[i] = (int32_t)((i * 7 + 3) % TABLE_ROWS);  // repeats on purpose
    for (uint32_t j = 0; j < WIDTH; ++j) {
      want[i * WIDTH + j] = table[(size_t)indices[i] * WIDTH + j];
    }
  }

  emunpu_tensor_id_t t = upload_f32(table, TABLE_ROWS * WIDTH, "emb.table");
  emunpu_tensor_id_t idx = alloc_n(COUNT, EMUNPU_DT_INT32, "emb.idx");
  emunpu_tensor_id_t c = alloc_n(COUNT * WIDTH, EMUNPU_DT_FP32, "emb.out");
  CHECK_OK(emunpu_device_upload(g_dev, idx, indices, sizeof(indices)));
  CHECK_OK(emunpu_gather(g_dev, NULL, c, t, idx, TABLE_ROWS, COUNT, WIDTH));
  download_f32(c, got, COUNT * WIDTH);
  check_array("gather", got, want, COUNT * WIDTH, 0.0);

  // An index past the end of the table must be caught, not read out of bounds.
  indices[5] = TABLE_ROWS;
  CHECK_OK(emunpu_device_upload(g_dev, idx, indices, sizeof(indices)));
  CHECK_ERR(emunpu_gather(g_dev, NULL, c, t, idx, TABLE_ROWS, COUNT, WIDTH));

  // A failed op must not leave queued work behind for the next one: run a
  // trivially checkable op right after the failure and require an exact result.
  {
    enum { N = 40 };
    float ones[N], out[N];
    for (uint32_t i = 0; i < N; ++i) ones[i] = 1.0f;
    emunpu_tensor_id_t a = upload_f32(ones, N, "after_err.a");
    emunpu_tensor_id_t o = alloc_n(N, EMUNPU_DT_FP32, "after_err.out");
    CHECK_OK(emunpu_vector_op(g_dev, NULL, o, a, a, N, EMUNPU_EW_ADD, 0));
    download_f32(o, out, N);
    int ok = 1;
    for (uint32_t i = 0; i < N; ++i)
      if (out[i] != 2.0f) ok = 0;
    CHECK(ok, "an op after a failed op produced wrong results");
    emunpu_device_free(g_dev, a);
    emunpu_device_free(g_dev, o);
  }

  emunpu_device_free(g_dev, t);
  emunpu_device_free(g_dev, idx);
  emunpu_device_free(g_dev, c);
}

//===----------------------------------------------------------------------===//
// Linear layer with a fused activation epilogue
//===----------------------------------------------------------------------===//

static void test_linear(void) {
  enum { M = 21, N = 19, K = 33 };  // all > MMA_DIM, none a multiple of it
  float x[M * K], weight[N * K], bias[N], got[M * N];
  double want[M * N];
  fill_rand(x, M * K, 51);
  fill_rand(weight, N * K, 52);
  fill_rand(bias, N, 53);

  emunpu_tensor_id_t dx = upload_f32(x, M * K, "lin.x");
  emunpu_tensor_id_t dw = upload_f32(weight, N * K, "lin.w");
  emunpu_tensor_id_t db = upload_f32(bias, N, "lin.b");
  emunpu_tensor_id_t dy = alloc_n(M * N, EMUNPU_DT_FP32, "lin.y");

  const emunpu_act_kind_t kinds[] = {EMUNPU_ACT_GELU, EMUNPU_ACT_SILU,
                                     EMUNPU_ACT_TANH};
  for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); ++k) {
    emunpu_act_t act = {kinds[k], 0.0, 0.0};
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < N; ++j) {
        double acc = bias[j];
        for (uint32_t p = 0; p < K; ++p)
          acc += (double)x[i * K + p] * weight[j * K + p];
        want[i * N + j] = ref_act(kinds[k], acc, 0.0, 0.0);
      }
    }
    // transpose_b=1: the weight is [N, K], as torch.nn.Linear stores it.
    CHECK_OK(emunpu_linear(g_dev, NULL, dy, dx, dw, db, M, N, K,
                           /*transpose_b=*/1, &act));
    download_f32(dy, got, M * N);
    char label[64];
    snprintf(label, sizeof(label), "linear + %s", emunpu_act_name(kinds[k]));
    check_array(label, got, want, M * N, 1e-5);
  }

  // No activation at all must equal plain gemm.
  {
    emunpu_tensor_id_t ref = alloc_n(M * N, EMUNPU_DT_FP32, "lin.ref");
    float from_gemm[M * N];
    CHECK_OK(emunpu_linear(g_dev, NULL, dy, dx, dw, db, M, N, K, 1, NULL));
    CHECK_OK(emunpu_gemm(g_dev, NULL, ref, dx, dw, db, M, N, K, 1, 0));
    download_f32(dy, got, M * N);
    download_f32(ref, from_gemm, M * N);
    int same = 1;
    for (uint32_t i = 0; i < M * N; ++i)
      if (got[i] != from_gemm[i]) same = 0;
    CHECK(same, "linear without an activation differs from gemm");
    emunpu_device_free(g_dev, ref);
  }

  emunpu_device_free(g_dev, dx);
  emunpu_device_free(g_dev, dw);
  emunpu_device_free(g_dev, db);
  emunpu_device_free(g_dev, dy);
}

//===----------------------------------------------------------------------===//
// Deferred execution: the new ops must work through a stream as well
//===----------------------------------------------------------------------===//

static void test_streamed_pipeline(void) {
  enum { ROWS = 12, WIDTH = 20, K = 17 };
  float x[ROWS * K], weight[WIDTH * K], got[ROWS * WIDTH];
  double want[ROWS * WIDTH];
  fill_rand(x, ROWS * K, 61);
  fill_rand(weight, WIDTH * K, 62);

  // Reference: softmax(gelu(x * Wt)) row-wise.
  for (uint32_t i = 0; i < ROWS; ++i) {
    double row[WIDTH];
    double max = -INFINITY, sum = 0.0;
    for (uint32_t j = 0; j < WIDTH; ++j) {
      double acc = 0.0;
      for (uint32_t p = 0; p < K; ++p)
        acc += (double)x[i * K + p] * weight[j * K + p];
      row[j] = ref_act(EMUNPU_ACT_GELU, acc, 0.0, 0.0);
      if (row[j] > max) max = row[j];
    }
    for (uint32_t j = 0; j < WIDTH; ++j) sum += exp(row[j] - max);
    for (uint32_t j = 0; j < WIDTH; ++j) {
      want[i * WIDTH + j] = exp(row[j] - max) / sum;
    }
  }

  emunpu_tensor_id_t dx = upload_f32(x, ROWS * K, "st.x");
  emunpu_tensor_id_t dw = upload_f32(weight, WIDTH * K, "st.w");
  emunpu_tensor_id_t dh = alloc_n(ROWS * WIDTH, EMUNPU_DT_FP32, "st.h");
  emunpu_stream_t* s = NULL;
  CHECK_OK(emunpu_stream_create(g_dev, &s));
  emunpu_act_t gelu = {EMUNPU_ACT_GELU, 0.0, 0.0};
  CHECK_OK(emunpu_linear(g_dev, s, dh, dx, dw, 0, ROWS, WIDTH, K, 1, &gelu));
  CHECK_OK(emunpu_softmax(g_dev, s, dh, dh, ROWS, WIDTH, 0));
  CHECK_OK(emunpu_device_synchronize(g_dev));
  download_f32(dh, got, ROWS * WIDTH);
  check_array("streamed linear+gelu+softmax", got, want, ROWS * WIDTH, 1e-5);
  emunpu_stream_destroy(s);

  // The same chain on a stream, including a conv2d whose scratch is released by
  // a queued FREE command rather than immediately.
  {
    emunpu_global_mem_t* gm = emunpu_device_global_mem(g_dev);
    uint64_t before = gm->bytes_in_use;
    emunpu_window_t w = {0};
    w.batch = 1;
    w.channels = 2;
    w.in_h = 6;
    w.in_w = 6;
    w.kernel_h = 3;
    w.kernel_w = 3;
    w.stride_h = w.stride_w = 1;
    w.pad_h = w.pad_w = 1;
    uint32_t oh = 0, ow = 0;
    emunpu_window_output(&w, &oh, &ow);
    uint32_t out_channels = 3;
    size_t in_elems = (size_t)w.channels * w.in_h * w.in_w;
    size_t k_elems = (size_t)out_channels * w.channels * 9;
    size_t out_elems = (size_t)out_channels * oh * ow;
    float* in = malloc(sizeof(float) * in_elems);
    float* weights = malloc(sizeof(float) * k_elems);
    float* conv_got = malloc(sizeof(float) * out_elems);
    double* conv_want = malloc(sizeof(double) * out_elems);
    fill_rand(in, in_elems, 71);
    fill_rand(weights, k_elems, 72);
    ref_conv2d(in, weights, NULL, conv_want, &w, out_channels, oh, ow,
               EMUNPU_ACT_RELU, 0);

    emunpu_tensor_id_t d_in = upload_f32(in, (uint32_t)in_elems, "cs.in");
    emunpu_tensor_id_t d_w = upload_f32(weights, (uint32_t)k_elems, "cs.w");
    emunpu_tensor_id_t d_out =
        alloc_n((uint32_t)out_elems, EMUNPU_DT_FP32, "cs.out");
    emunpu_stream_t* cs = NULL;
    CHECK_OK(emunpu_stream_create(g_dev, &cs));
    CHECK_OK(
        emunpu_conv2d(g_dev, cs, d_out, d_in, d_w, 0, &w, out_channels, NULL));
    CHECK_OK(emunpu_stream_synchronize(cs));
    download_f32(d_out, conv_got, (uint32_t)out_elems);
    check_array("streamed conv2d", conv_got, conv_want, (uint32_t)out_elems,
                1e-5);
    emunpu_stream_destroy(cs);
    emunpu_device_free(g_dev, d_in);
    emunpu_device_free(g_dev, d_w);
    emunpu_device_free(g_dev, d_out);
    CHECK(gm->bytes_in_use == before,
          "streamed conv2d leaked global memory (%llu -> %llu bytes)",
          (unsigned long long)before, (unsigned long long)gm->bytes_in_use);
    free(in);
    free(weights);
    free(conv_got);
    free(conv_want);
  }

  emunpu_device_free(g_dev, dx);
  emunpu_device_free(g_dev, dw);
  emunpu_device_free(g_dev, dh);
}

//===----------------------------------------------------------------------===//
// Hardware capabilities and per-instruction limits
//===----------------------------------------------------------------------===//

static void test_capabilities(void) {
  emunpu_caps_t caps;
  CHECK_OK(emunpu_device_query_caps(g_dev, &caps));
  CHECK(caps.num_cores == emunpu_device_num_cores(g_dev),
        "caps.num_cores %u disagrees with emunpu_device_num_cores %u",
        caps.num_cores, emunpu_device_num_cores(g_dev));
  CHECK(caps.mma_dim == IREE_EMUNPU_MMA_DIM, "caps.mma_dim = %u", caps.mma_dim);
  CHECK(caps.vector_lanes == IREE_EMUNPU_VECTOR_LANES, "caps.vector_lanes = %u",
        caps.vector_lanes);
  CHECK(caps.sfu_lanes == IREE_EMUNPU_SFU_LANES, "caps.sfu_lanes = %u",
        caps.sfu_lanes);
  CHECK(caps.max_row_width == IREE_EMUNPU_MAX_ROW_WIDTH,
        "caps.max_row_width = %u", caps.max_row_width);
  CHECK(caps.dma_max_bytes == IREE_EMUNPU_DMA_MAX_BYTES,
        "caps.dma_max_bytes = %llu", (unsigned long long)caps.dma_max_bytes);
  CHECK(caps.local_mem_bytes > 0 && caps.global_mem_bytes > 0,
        "caps memory sizes must be non-zero");
  CHECK(caps.dtype_count == EMUNPU_DT_COUNT, "caps.dtype_count = %u",
        caps.dtype_count);
  CHECK(emunpu_device_query_caps(NULL, &caps) != EMUNPU_OK,
        "query_caps(NULL) should fail");
}

// A single instruction may not ask for more work than its unit can do in one
// issue. Each case here is "the largest legal instruction is accepted, one
// element more is rejected" -- so the limit is exact, not approximate.
static void test_instruction_limits(void) {
  emunpu_caps_t caps;
  CHECK_OK(emunpu_device_query_caps(g_dev, &caps));

  struct {
    const char* name;
    emunpu_iopcode_t op;
    uint32_t at_limit_M, at_limit_N;  // largest legal (M, N)
    uint32_t over_M, over_N;          // one step past the limit
  } cases[] = {
      {"EW (vector lanes)", EMUNPU_I_EW, caps.vector_lanes, 0,
       caps.vector_lanes + 1, 0},
      {"FMA (vector lanes)", EMUNPU_I_FMA, caps.vector_lanes, 0,
       caps.vector_lanes + 1, 0},
      {"CONVERT (vector lanes)", EMUNPU_I_CONVERT, caps.vector_lanes, 0,
       caps.vector_lanes + 1, 0},
      {"ACT (sfu lanes)", EMUNPU_I_ACT, caps.sfu_lanes, 0, caps.sfu_lanes + 1,
       0},
      {"MMA16 (mma dim)", EMUNPU_I_MMA16, caps.mma_dim, caps.mma_dim,
       caps.mma_dim + 1, caps.mma_dim},
      {"MATVEC (mma dim)", EMUNPU_I_MATVEC, caps.mma_dim, 0, caps.mma_dim + 1,
       0},
      {"TRANSPOSE (mma dim)", EMUNPU_I_TRANSPOSE, caps.mma_dim, caps.mma_dim,
       caps.mma_dim, caps.mma_dim + 1},
      {"GATHER (rows)", EMUNPU_I_GATHER, caps.gather_rows, 8,
       caps.gather_rows + 1, 8},
      {"IM2COL (rows)", EMUNPU_I_IM2COL, caps.im2col_rows, 8,
       caps.im2col_rows + 1, 8},
      {"SOFTMAX (one row)", EMUNPU_I_SOFTMAX, 1, caps.max_row_width, 2,
       caps.max_row_width},
      {"NORM (row width)", EMUNPU_I_NORM, 1, caps.max_row_width, 1,
       caps.max_row_width + 1},
      {"REDUCE (row width)", EMUNPU_I_REDUCE, 1, caps.max_row_width, 1,
       caps.max_row_width + 1},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    emunpu_instr_t in = {0};
    in.op = cases[i].op;
    in.a_dtype = in.b_dtype = in.c_dtype = EMUNPU_DT_FP32;
    in.M = cases[i].at_limit_M;
    in.N = cases[i].at_limit_N;
    if (cases[i].op == EMUNPU_I_MMA16 || cases[i].op == EMUNPU_I_MATVEC) {
      in.K = caps.mma_dim;
    }
    CHECK(emunpu_instr_validate(&in) == EMUNPU_OK,
          "%s: the largest legal instruction was rejected", cases[i].name);
    in.M = cases[i].over_M;
    in.N = cases[i].over_N;
    CHECK(emunpu_instr_validate(&in) == EMUNPU_ERROR_OUT_OF_RANGE,
          "%s: an oversized instruction was accepted", cases[i].name);
    // The same check must happen at enqueue, not just in the validator.
    CHECK_ERR(emunpu_device_enqueue(g_dev, 0, &in));
  }

  // A DMA descriptor is bounded in bytes, not elements.
  {
    emunpu_instr_t in = {0};
    in.op = EMUNPU_I_DMA_G2L;
    in.c_dtype = EMUNPU_DT_FP32;
    in.M = 1;
    in.N = (uint32_t)(caps.dma_max_bytes / sizeof(float));
    CHECK(emunpu_instr_validate(&in) == EMUNPU_OK,
          "a full-size DMA descriptor was rejected");
    in.N += 1;
    CHECK(emunpu_instr_validate(&in) == EMUNPU_ERROR_OUT_OF_RANGE,
          "an oversized DMA descriptor was accepted");
  }

  // An unknown opcode is not silently ignored.
  {
    emunpu_instr_t in = {0};
    in.op = (emunpu_iopcode_t)0;
    CHECK(emunpu_instr_validate(&in) == EMUNPU_ERROR_UNIMPLEMENTED,
          "opcode 0 should be unimplemented");
  }
  CHECK(emunpu_instr_validate(NULL) != EMUNPU_OK, "validate(NULL) should fail");

  // The device layer tiles to these limits, so tensor-sized work is fine -- but
  // a row wider than the row unit's buffer cannot be expressed at all, and must
  // be reported rather than silently truncated.
  {
    uint32_t wide = caps.max_row_width + 1;
    emunpu_tensor_id_t a = alloc_n(wide, EMUNPU_DT_FP32, "wide.in");
    emunpu_tensor_id_t c = alloc_n(wide, EMUNPU_DT_FP32, "wide.out");
    CHECK_ERR(emunpu_softmax(g_dev, NULL, c, a, 1, wide, 0));
    CHECK_ERR(emunpu_layernorm(g_dev, NULL, c, a, 0, 0, 1, wide, 1e-5));
    CHECK_ERR(emunpu_reduce(g_dev, NULL, c, a, 1, wide, EMUNPU_REDUCE_SUM));
    // ...while an elementwise op of the same size just becomes several
    // instructions and succeeds.
    CHECK_OK(emunpu_vector_op(g_dev, NULL, c, a, 0, wide, EMUNPU_EW_COPY, 0));
    emunpu_device_free(g_dev, a);
    emunpu_device_free(g_dev, c);
  }
}

//===----------------------------------------------------------------------===//
// Execution scope: choosing how many cores an operation uses
//===----------------------------------------------------------------------===//

static void test_exec_scope(void) {
  enum { M = 40, N = 48, K = 24 };
  float a[M * K], b[K * N], full[M * N], scoped[M * N];
  fill_rand(a, M * K, 81);
  fill_rand(b, K * N, 82);
  emunpu_tensor_id_t da = upload_f32(a, M * K, "scope.a");
  emunpu_tensor_id_t db = upload_f32(b, K * N, "scope.b");
  emunpu_tensor_id_t dc = alloc_n(M * N, EMUNPU_DT_FP32, "scope.c");

  uint32_t cores = emunpu_device_num_cores(g_dev);
  uint32_t first = 0, count = 0;
  emunpu_device_get_exec_scope(g_dev, &first, &count);
  CHECK(first == 0 && count == cores,
        "the default scope should be every core, got [%u, %u)", first,
        first + count);

  // Baseline: the whole device.
  CHECK_OK(emunpu_matmul(g_dev, NULL, dc, da, db, M, N, K, 0));
  download_f32(dc, full, M * N);

  // A narrowed scope must produce identical results and use only its cores.
  uint32_t scope_first = cores / 2;
  uint32_t scope_count = cores - scope_first;
  emunpu_core_stats_t before[IREE_EMUNPU_NUM_CORES];
  emunpu_core_stats_t after[IREE_EMUNPU_NUM_CORES];
  for (uint32_t i = 0; i < cores; ++i) {
    CHECK_OK(emunpu_device_core_stats(g_dev, i, &before[i]));
  }
  CHECK_OK(emunpu_device_set_exec_scope(g_dev, scope_first, scope_count));
  emunpu_device_get_exec_scope(g_dev, &first, &count);
  CHECK(first == scope_first && count == scope_count,
        "scope readback mismatch: [%u, %u)", first, first + count);
  CHECK_OK(emunpu_matmul(g_dev, NULL, dc, da, db, M, N, K, 0));
  download_f32(dc, scoped, M * N);
  for (uint32_t i = 0; i < cores; ++i) {
    CHECK_OK(emunpu_device_core_stats(g_dev, i, &after[i]));
  }

  int same = 1;
  for (uint32_t i = 0; i < M * N; ++i)
    if (full[i] != scoped[i]) same = 0;
  CHECK(same, "narrowing the execution scope changed the result");

  uint32_t worked = 0, trespassed = 0;
  for (uint32_t i = 0; i < cores; ++i) {
    uint64_t delta = after[i].instr_count - before[i].instr_count;
    int in_scope = (i >= scope_first && i < scope_first + scope_count);
    if (in_scope && delta) ++worked;
    if (!in_scope && delta) ++trespassed;
  }
  CHECK(trespassed == 0, "%u cores outside the scope executed work",
        trespassed);
  CHECK(worked > 0, "no core inside the scope executed work");

  // Pinning to a single core must also work, and still be correct.
  CHECK_OK(emunpu_device_set_exec_scope(g_dev, cores - 1, 1));
  CHECK_OK(emunpu_matmul(g_dev, NULL, dc, da, db, M, N, K, 0));
  download_f32(dc, scoped, M * N);
  same = 1;
  for (uint32_t i = 0; i < M * N; ++i)
    if (full[i] != scoped[i]) same = 0;
  CHECK(same, "a single-core scope changed the result");

  // Nonsense scopes are rejected.
  CHECK_ERR(emunpu_device_set_exec_scope(g_dev, 0, 0));
  CHECK_ERR(emunpu_device_set_exec_scope(g_dev, 0, cores + 1));
  CHECK_ERR(emunpu_device_set_exec_scope(g_dev, cores, 1));

  emunpu_device_reset_exec_scope(g_dev);
  emunpu_device_get_exec_scope(g_dev, &first, &count);
  CHECK(first == 0 && count == cores, "reset_exec_scope did not restore");

  emunpu_device_free(g_dev, da);
  emunpu_device_free(g_dev, db);
  emunpu_device_free(g_dev, dc);
}

//===----------------------------------------------------------------------===//
// Low-level programming: a hand-written tiled GEMM
//
// No high-level op is used here. The test queries the hardware's limits, blocks
// the problem itself, stages every tile through the cores' local scratchpads
// with explicit DMA, issues MMA instructions against local memory, and writes
// the result back -- i.e. exactly what a code generator emits.
//===----------------------------------------------------------------------===//

static void test_raw_programming(void) {
  enum { M = 40, N = 48, K = 24 };
  emunpu_caps_t caps;
  CHECK_OK(emunpu_device_query_caps(g_dev, &caps));
  const uint32_t tile = caps.mma_dim;

  float a[M * K], b[K * N], got[M * N];
  double want[M * N];
  fill_rand(a, M * K, 91);
  fill_rand(b, K * N, 92);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k)
        acc += (double)a[i * K + k] * b[k * N + j];
      want[i * N + j] = acc;
    }
  }

  emunpu_tensor_id_t da = upload_f32(a, M * K, "raw.a");
  emunpu_tensor_id_t db = upload_f32(b, K * N, "raw.b");
  emunpu_tensor_id_t dc = alloc_n(M * N, EMUNPU_DT_FP32, "raw.c");

  // Operand base addresses in the global address space.
  emunpu_global_mem_t* gm = emunpu_device_global_mem(g_dev);
  uint64_t a_base = 0, b_base = 0, c_base = 0;
  CHECK_OK(emunpu_global_mem_offset(gm, da, &a_base));
  CHECK_OK(emunpu_global_mem_offset(gm, db, &b_base));
  CHECK_OK(emunpu_global_mem_offset(gm, dc, &c_base));

  // One A/B/C tile buffer per core in its scratchpad, reused by every tile that
  // core handles (its instructions run in order, so reuse is safe).
  const uint64_t tile_bytes = (uint64_t)tile * tile * sizeof(float);
  uint64_t a_local[IREE_EMUNPU_NUM_CORES], b_local[IREE_EMUNPU_NUM_CORES],
      c_local[IREE_EMUNPU_NUM_CORES];
  for (uint32_t core = 0; core < caps.num_cores; ++core) {
    CHECK_OK(emunpu_device_local_reset(g_dev, core));
    CHECK_OK(
        emunpu_device_local_alloc(g_dev, core, tile_bytes, &a_local[core]));
    CHECK_OK(
        emunpu_device_local_alloc(g_dev, core, tile_bytes, &b_local[core]));
    CHECK_OK(
        emunpu_device_local_alloc(g_dev, core, tile_bytes, &c_local[core]));
    CHECK(a_local[core] != b_local[core] && b_local[core] != c_local[core],
          "scratchpad allocations overlap on core %u", core);
  }
  // The scratchpad is finite: asking for more than it holds must fail.
  CHECK_ERR(emunpu_device_local_alloc(g_dev, 0, caps.local_mem_bytes + 1,
                                      &a_local[0]));
  CHECK_ERR(emunpu_device_local_alloc(g_dev, caps.num_cores, tile_bytes,
                                      &a_local[0]));

  uint32_t tile_index = 0;
  for (uint32_t m0 = 0; m0 < M; m0 += tile) {
    uint32_t tm = (M - m0 < tile) ? (M - m0) : tile;
    for (uint32_t n0 = 0; n0 < N; n0 += tile, ++tile_index) {
      uint32_t tn = (N - n0 < tile) ? (N - n0) : tile;
      uint32_t core = tile_index % caps.num_cores;  // our own core assignment
      for (uint32_t k0 = 0; k0 < K; k0 += tile) {
        uint32_t tk = (K - k0 < tile) ? (K - k0) : tile;
        // A[m0:m0+tm, k0:k0+tk] global -> local, packed.
        emunpu_instr_t dma_a = {0};
        dma_a.op = EMUNPU_I_DMA_G2L;
        dma_a.a_dtype = dma_a.c_dtype = EMUNPU_DT_FP32;
        dma_a.M = tm;
        dma_a.N = tk;
        dma_a.a_offset = a_base + ((uint64_t)m0 * K + k0) * sizeof(float);
        dma_a.a_row_stride = K * sizeof(float);
        dma_a.c_offset = a_local[core];
        CHECK_OK(emunpu_device_enqueue(g_dev, core, &dma_a));

        // B[k0:k0+tk, n0:n0+tn] global -> local, packed.
        emunpu_instr_t dma_b = dma_a;
        dma_b.M = tk;
        dma_b.N = tn;
        dma_b.a_offset = b_base + ((uint64_t)k0 * N + n0) * sizeof(float);
        dma_b.a_row_stride = N * sizeof(float);
        dma_b.c_offset = b_local[core];
        CHECK_OK(emunpu_device_enqueue(g_dev, core, &dma_b));

        // C_tile (+)= A_tile * B_tile, entirely in local memory.
        emunpu_instr_t mma = {0};
        mma.op = EMUNPU_I_MMA16;
        mma.a_dtype = mma.b_dtype = mma.acc_dtype = mma.c_dtype =
            EMUNPU_DT_FP32;
        mma.M = tm;
        mma.N = tn;
        mma.K = tk;
        mma.a_space = mma.b_space = mma.c_space = EMUNPU_SPACE_LOCAL;
        mma.a_offset = a_local[core];
        mma.b_offset = b_local[core];
        mma.c_offset = c_local[core];
        mma.flags = k0 ? EMUNPU_MMA_ACCUMULATE : 0;
        CHECK_OK(emunpu_device_enqueue(g_dev, core, &mma));
      }
      // The finished C tile local -> global.
      emunpu_instr_t dma_c = {0};
      dma_c.op = EMUNPU_I_DMA_L2G;
      dma_c.a_dtype = dma_c.c_dtype = EMUNPU_DT_FP32;
      dma_c.M = tm;
      dma_c.N = tn;
      dma_c.a_offset = c_local[core];
      dma_c.c_offset = c_base + ((uint64_t)m0 * N + n0) * sizeof(float);
      dma_c.c_row_stride = N * sizeof(float);
      CHECK_OK(emunpu_device_enqueue(g_dev, core, &dma_c));
    }
  }

  // Nothing has executed yet -- the instructions are sitting in the queues.
  emunpu_core_stats_t queued;
  CHECK_OK(emunpu_device_core_stats(g_dev, 0, &queued));
  CHECK(queued.queue_len > 0, "core 0 should have queued instructions");

  CHECK_OK(emunpu_device_launch(g_dev));
  download_f32(dc, got, M * N);
  check_array("hand-scheduled tiled GEMM", got, want, M * N, 1e-5);

  emunpu_core_stats_t drained;
  CHECK_OK(emunpu_device_core_stats(g_dev, 0, &drained));
  CHECK(drained.queue_len == 0, "the queue should be empty after a launch");
  CHECK(drained.mma_count > queued.mma_count, "core 0 executed no MMA");
  CHECK(drained.dma_bytes > queued.dma_bytes, "core 0 moved no DMA bytes");

  for (uint32_t core = 0; core < caps.num_cores; ++core) {
    CHECK_OK(emunpu_device_local_reset(g_dev, core));
  }
  emunpu_device_free(g_dev, da);
  emunpu_device_free(g_dev, db);
  emunpu_device_free(g_dev, dc);
}

//===----------------------------------------------------------------------===//
// Matrix-vector (its lowering blocks both M and K across the matrix engine)
//===----------------------------------------------------------------------===//

static void test_matvec(void) {
  enum { M = 37, K = 29 };  // both > MMA_DIM and not multiples of it
  float a[M * K], x[K], got[M];
  double want[M];
  fill_rand(a, M * K, 101);
  fill_rand(x, K, 102);
  for (uint32_t i = 0; i < M; ++i) {
    double acc = 0.0;
    for (uint32_t k = 0; k < K; ++k) acc += (double)a[i * K + k] * x[k];
    want[i] = acc;
  }
  emunpu_tensor_id_t da = upload_f32(a, M * K, "mv.a");
  emunpu_tensor_id_t dx = upload_f32(x, K, "mv.x");
  emunpu_tensor_id_t dy = alloc_n(M, EMUNPU_DT_FP32, "mv.y");
  CHECK_OK(emunpu_matvec(g_dev, NULL, dy, da, dx, M, K));
  download_f32(dy, got, M);
  check_array("matvec", got, want, M, 1e-5);
  emunpu_device_free(g_dev, da);
  emunpu_device_free(g_dev, dx);
  emunpu_device_free(g_dev, dy);
}

//===----------------------------------------------------------------------===//

int main(void) {
  CHECK_OK(emunpu_device_create(0, &g_dev));
  if (!g_dev) {
    fprintf(stderr, "could not create the device\n");
    return 1;
  }
  printf("emunpu ops test: %u cores\n", emunpu_device_num_cores(g_dev));

  test_capabilities();
  test_instruction_limits();
  test_exec_scope();
  test_raw_programming();
  test_activations();
  test_reductions();
  test_matvec();
  test_ew_pow();
  test_softmax();
  test_norm();
  test_transpose();
  test_im2col();
  test_conv2d();
  test_pool2d();
  test_gather();
  test_linear();
  test_streamed_pipeline();

  emunpu_device_destroy(g_dev);
  printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
