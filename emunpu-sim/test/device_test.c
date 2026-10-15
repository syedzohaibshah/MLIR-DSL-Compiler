// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Standalone functional test for the multi-core NPU simulator.
//
// Builds with nothing but the C standard library + the simulator sources, so it
// can be run without any external dependency:
//
//   make test                      # or:
//   cc -Iinclude -lm src/*.c test/device_test.c -o emunpu_test
//===----------------------------------------------------------------------===//

#include "emunpu/device.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

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

#define CHECK_OK(expr) CHECK((expr) == EMUNPU_OK, "expected OK from " #expr)

// Fills a host buffer of |dtype| with values from a simple deterministic
// generator (small integers, exactly representable in every supported dtype).
static void fill_dtype(void* buf, uint32_t n, emunpu_dtype_t dtype, int seed) {
  for (uint32_t i = 0; i < n; ++i) {
    double v = (double)(((int)i * 7 + seed * 3) % 5 - 2);  // in {-2..2}
    emunpu_dtype_store(buf, i, dtype, v);
  }
}

// Reference GEMM in double using values read back through the dtype accessors,
// so it matches the device's rounding for exactly-representable inputs.
static void ref_gemm(const void* A, const void* B, double* C, uint32_t M,
                     uint32_t N, uint32_t K, emunpu_dtype_t adt,
                     emunpu_dtype_t bdt, int transpose_b) {
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        double a = emunpu_dtype_load(A, i * K + k, adt);
        double b = transpose_b ? emunpu_dtype_load(B, j * K + k, bdt)
                               : emunpu_dtype_load(B, k * N + j, bdt);
        acc += a * b;
      }
      C[i * N + j] = acc;
    }
  }
}

static void test_gemm_dtype(emunpu_device_t* dev, uint32_t M, uint32_t N,
                            uint32_t K, emunpu_dtype_t adt, emunpu_dtype_t bdt,
                            emunpu_dtype_t cdt, int transpose_b,
                            const char* label) {
  size_t as = emunpu_dtype_size(adt), bs = emunpu_dtype_size(bdt),
         cs = emunpu_dtype_size(cdt);
  void* hA = malloc((size_t)M * K * as);
  void* hB = malloc((size_t)K * N * bs);
  void* hC = malloc((size_t)M * N * cs);
  fill_dtype(hA, M * K, adt, 1);
  fill_dtype(hB, K * N, bdt, 2);

  emunpu_tensor_id_t a, b, c;
  uint32_t sa[2] = {M, K}, sbN[2] = {K, N}, sbT[2] = {N, K}, sc[2] = {M, N};
  CHECK_OK(emunpu_device_alloc(dev, adt, 2, sa, "A", &a));
  CHECK_OK(emunpu_device_alloc(dev, bdt, 2, transpose_b ? sbT : sbN, "B", &b));
  CHECK_OK(emunpu_device_alloc(dev, cdt, 2, sc, "C", &c));
  CHECK_OK(emunpu_device_upload(dev, a, hA, (uint64_t)M * K * as));
  CHECK_OK(emunpu_device_upload(dev, b, hB, (uint64_t)K * N * bs));
  CHECK_OK(emunpu_matmul(dev, NULL, c, a, b, M, N, K, transpose_b));
  CHECK_OK(emunpu_device_download(dev, c, hC, (uint64_t)M * N * cs));

  double* ref = malloc((size_t)M * N * sizeof(double));
  ref_gemm(hA, hB, ref, M, N, K, adt, bdt, transpose_b);
  int ok = 1;
  for (uint32_t i = 0; i < M * N; ++i) {
    double got = emunpu_dtype_load(hC, i, cdt);
    if (got != ref[i]) {
      ok = 0;
      if (i < 4)
        fprintf(stderr, "  [%s] C[%u] got %g want %g\n", label, i, got, ref[i]);
    }
  }
  CHECK(ok, "GEMM %s (%ux%ux%u transpose_b=%d) mismatch", label, M, N, K,
        transpose_b);

  emunpu_device_free(dev, a);
  emunpu_device_free(dev, b);
  emunpu_device_free(dev, c);
  free(hA);
  free(hB);
  free(hC);
  free(ref);
}

int main(void) {
  emunpu_device_t* dev = NULL;
  CHECK_OK(emunpu_device_create(0, &dev));
  // How many cores come online is a runtime knob (IREE_EMUNPU_NUM_CORES,
  // clamped to the physical count), so check against what was actually
  // requested rather than against a constant -- otherwise this test cannot run
  // on a reduced core count, which is exactly one of the configurations worth
  // testing.
  uint32_t expected_cores = IREE_EMUNPU_NUM_CORES;
  const char* core_env = getenv("IREE_EMUNPU_NUM_CORES");
  if (core_env && core_env[0] != '\0') {
    long n = strtol(core_env, NULL, 10);
    if (n < 1) n = 1;
    if (n > IREE_EMUNPU_NUM_CORES) n = IREE_EMUNPU_NUM_CORES;
    expected_cores = (uint32_t)n;
  }
  CHECK(emunpu_device_num_cores(dev) == expected_cores,
        "expected %u cores, got %u", expected_cores,
        emunpu_device_num_cores(dev));

  // --- GEMM: large, tiled, ragged edges, multi-core, fp32 std + transpose_b.
  test_gemm_dtype(dev, 40, 48, 24, EMUNPU_DT_FP32, EMUNPU_DT_FP32,
                  EMUNPU_DT_FP32, 0, "fp32 std");
  test_gemm_dtype(dev, 33, 17, 40, EMUNPU_DT_FP32, EMUNPU_DT_FP32,
                  EMUNPU_DT_FP32, 1, "fp32 transpose_b");

  // --- All datatypes (small splat-friendly sizes, values exact in each dtype).
  emunpu_dtype_t dts[] = {EMUNPU_DT_FP32,  EMUNPU_DT_FP16,  EMUNPU_DT_BF16,
                          EMUNPU_DT_INT32, EMUNPU_DT_INT16, EMUNPU_DT_INT8,
                          EMUNPU_DT_UINT8};
  for (size_t i = 0; i < sizeof(dts) / sizeof(dts[0]); ++i) {
    test_gemm_dtype(dev, 18, 20, 12, dts[i], dts[i], dts[i], 0,
                    emunpu_dtype_name(dts[i]));
  }

  // --- Mixed precision: fp16 inputs, fp32 accumulate/output; i8 inputs, i32
  // out.
  test_gemm_dtype(dev, 20, 20, 20, EMUNPU_DT_FP16, EMUNPU_DT_FP16,
                  EMUNPU_DT_FP32, 0, "fp16->fp32");
  test_gemm_dtype(dev, 20, 20, 20, EMUNPU_DT_BF16, EMUNPU_DT_BF16,
                  EMUNPU_DT_FP32, 0, "bf16->fp32");
  test_gemm_dtype(dev, 17, 19, 16, EMUNPU_DT_INT8, EMUNPU_DT_INT8,
                  EMUNPU_DT_INT32, 0, "i8->i32");

  // --- Vector ops + broadcast bias add.
  {
    uint32_t n = 100, shp = n;
    float ha[100], hb[100], hc[100];
    for (uint32_t i = 0; i < n; ++i) {
      ha[i] = (float)i;
      hb[i] = (float)(2 * i);
    }
    emunpu_tensor_id_t a, b, c;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "a", &a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "b", &b));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "c", &c));
    CHECK_OK(emunpu_device_upload(dev, a, ha, sizeof(ha)));
    CHECK_OK(emunpu_device_upload(dev, b, hb, sizeof(hb)));
    CHECK_OK(emunpu_vector_op(dev, NULL, c, a, b, n, EMUNPU_EW_ADD, 0));
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    int ok = 1;
    for (uint32_t i = 0; i < n; ++i)
      if (hc[i] != ha[i] + hb[i]) ok = 0;
    CHECK(ok, "vector add mismatch");
    CHECK_OK(emunpu_vector_op(dev, NULL, c, a, 0, n, EMUNPU_EW_RELU, 0));
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    ok = 1;
    for (uint32_t i = 0; i < n; ++i)
      if (hc[i] != (ha[i] > 0 ? ha[i] : 0)) ok = 0;
    CHECK(ok, "vector relu mismatch");
    emunpu_device_free(dev, a);
    emunpu_device_free(dev, b);
    emunpu_device_free(dev, c);
  }

  // --- Reduction (sum over rows).
  {
    uint32_t rows = 7, width = 13;
    uint32_t sa[2] = {rows, width}, sc = rows;
    float* ha = malloc(sizeof(float) * rows * width);
    float hc[7];
    for (uint32_t i = 0; i < rows * width; ++i) ha[i] = (float)(i % 9);
    emunpu_tensor_id_t a, c;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 2, sa, "a", &a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &sc, "c", &c));
    CHECK_OK(emunpu_device_upload(dev, a, ha, sizeof(float) * rows * width));
    CHECK_OK(emunpu_reduce(dev, NULL, c, a, rows, width, EMUNPU_REDUCE_SUM));
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    int ok = 1;
    for (uint32_t r = 0; r < rows; ++r) {
      double s = 0;
      for (uint32_t j = 0; j < width; ++j) s += ha[r * width + j];
      if (hc[r] != (float)s) ok = 0;
    }
    CHECK(ok, "reduction sum mismatch");
    emunpu_device_free(dev, a);
    emunpu_device_free(dev, c);
    free(ha);
  }

  // --- FMA.
  {
    uint32_t n = 64, shp = n;
    float ha[64], hb[64], hc[64], expect[64];
    for (uint32_t i = 0; i < n; ++i) {
      ha[i] = (float)i;
      hb[i] = 3.0f;
      hc[i] = 1.0f;
      expect[i] = ha[i] * hb[i] + hc[i];
    }
    emunpu_tensor_id_t a, b, c;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "a", &a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "b", &b));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "c", &c));
    CHECK_OK(emunpu_device_upload(dev, a, ha, sizeof(ha)));
    CHECK_OK(emunpu_device_upload(dev, b, hb, sizeof(hb)));
    CHECK_OK(emunpu_device_upload(dev, c, hc, sizeof(hc)));
    CHECK_OK(emunpu_fma(dev, NULL, c, a, b, n));
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    int ok = 1;
    for (uint32_t i = 0; i < n; ++i)
      if (hc[i] != expect[i]) ok = 0;
    CHECK(ok, "fma mismatch");
    emunpu_device_free(dev, a);
    emunpu_device_free(dev, b);
    emunpu_device_free(dev, c);
  }

  // --- Datatype conversion fp32 -> fp16 -> fp32 round trip (exact for ints).
  {
    uint32_t n = 32, shp = n;
    float ha[32], hc[32];
    for (uint32_t i = 0; i < n; ++i) ha[i] = (float)((int)i - 16);
    emunpu_tensor_id_t a, h, c;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "a", &a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP16, 1, &shp, "h", &h));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "c", &c));
    CHECK_OK(emunpu_device_upload(dev, a, ha, sizeof(ha)));
    CHECK_OK(emunpu_convert(dev, NULL, h, a, n));  // fp32 -> fp16
    CHECK_OK(emunpu_convert(dev, NULL, c, h, n));  // fp16 -> fp32
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    int ok = 1;
    for (uint32_t i = 0; i < n; ++i)
      if (hc[i] != ha[i]) ok = 0;
    CHECK(ok, "convert round-trip mismatch");
    emunpu_device_free(dev, a);
    emunpu_device_free(dev, h);
    emunpu_device_free(dev, c);
  }

  // --- Streams + events + barrier: deferred work executes on synchronize.
  {
    uint32_t n = 50, shp = n;
    float ha[50], hb[50], hc[50];
    for (uint32_t i = 0; i < n; ++i) {
      ha[i] = (float)i;
      hb[i] = 10.0f;
    }
    emunpu_tensor_id_t a, b, c;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "a", &a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "b", &b));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "c", &c));
    CHECK_OK(emunpu_device_upload(dev, a, ha, sizeof(ha)));
    CHECK_OK(emunpu_device_upload(dev, b, hb, sizeof(hb)));
    emunpu_stream_t* s = NULL;
    emunpu_event_t* e = NULL;
    CHECK_OK(emunpu_stream_create(dev, &s));
    CHECK_OK(emunpu_event_create(dev, &e));
    CHECK_OK(emunpu_vector_op(dev, s, c, a, b, n, EMUNPU_EW_ADD, 0));
    CHECK_OK(
        emunpu_vector_op(dev, s, c, c, b, n, EMUNPU_EW_MUL, 0));  // (a+b)*b
    CHECK_OK(emunpu_event_record(e, s));
    // Nothing has run yet on the host's view until we synchronize.
    CHECK_OK(emunpu_event_synchronize(e));
    CHECK_OK(emunpu_device_barrier(dev));
    CHECK_OK(emunpu_device_download(dev, c, hc, sizeof(hc)));
    int ok = 1;
    for (uint32_t i = 0; i < n; ++i)
      if (hc[i] != (ha[i] + hb[i]) * hb[i]) ok = 0;
    CHECK(ok, "stream/event pipeline mismatch");
    emunpu_event_destroy(e);
    emunpu_stream_destroy(s);
    emunpu_device_free(dev, a);
    emunpu_device_free(dev, b);
    emunpu_device_free(dev, c);
  }

  // --- Global memory free-list reuse: free then re-alloc reuses the region.
  {
    emunpu_global_mem_t* gm = emunpu_device_global_mem(dev);
    uint32_t shp = 256;
    emunpu_tensor_id_t a, b;
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "a", &a));
    uint64_t off_a = 0;
    CHECK_OK(emunpu_global_mem_offset(gm, a, &off_a));
    CHECK_OK(emunpu_device_free(dev, a));
    CHECK_OK(emunpu_device_alloc(dev, EMUNPU_DT_FP32, 1, &shp, "b", &b));
    uint64_t off_b = 0;
    CHECK_OK(emunpu_global_mem_offset(gm, b, &off_b));
    CHECK(off_a == off_b, "freed region not reused (%llu vs %llu)",
          (unsigned long long)off_a, (unsigned long long)off_b);
    emunpu_device_free(dev, b);
  }

  emunpu_device_destroy(dev);

  printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
