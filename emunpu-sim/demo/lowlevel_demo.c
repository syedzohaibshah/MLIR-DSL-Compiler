// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Driving the NPU at the instruction level.
//
// The other demos call whole-tensor operations and let the device layer tile
// and distribute the work. This one goes underneath that, the way a code
// generator would, and shows the three things the low-level API is for:
//
//   1. Query the hardware. How many cores, how much memory, and -- the part
//   that
//      actually constrains a schedule -- how much work ONE instruction of each
//      functional unit may do (emunpu_device_query_caps).
//
//   2. Hand-write a kernel. A tiled GEMM built from raw instructions: block the
//      problem to the matrix engine's tile size, assign tiles to cores
//      yourself, DMA each operand tile into the core's local scratchpad, issue
//      MMA against local memory, accumulate over K, DMA the result back. No
//      high-level op is involved, and the per-core counters afterwards show
//      exactly what each core did.
//
//   3. Choose how many cores an operation uses. The same GEMM is then run
//      through the high-level API under execution scopes of 1, 2, ... cores, so
//      you can see the work spread out (and confirm the results are identical
//      however it is partitioned).
//
//   # watch it run (two terminals):
//   python demo/visualizer.py /tmp/emunpu.events
//   IREE_EMUNPU_VIZ=/tmp/emunpu.events ./lowlevel_demo
//
// IREE_EMUNPU_TRACE=1 logs every instruction the cores execute.
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "emunpu/device.h"

// Deliberately not multiples of the 16x16 tile, so the edge tiles are partial.
enum { M = 72, N = 56, K = 40 };

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

static double max_abs_err(const float* got, const double* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double e = fabs((double)got[i] - want[i]);
    if (e > worst) worst = e;
  }
  return worst;
}

int main(void) {
  emunpu_device_t* dev = NULL;
  CHECK_OK(emunpu_device_create(0, &dev));

  //=== 1. What can this hardware do? ======================================
  emunpu_caps_t caps;
  CHECK_OK(emunpu_device_query_caps(dev, &caps));
  printf("emunpu hardware capabilities\n");
  printf("  cores                    %u\n", caps.num_cores);
  printf("  global memory            %.1f MB\n",
         (double)caps.global_mem_bytes / (1024.0 * 1024.0));
  printf("  local scratchpad / core   %.0f KB\n",
         (double)caps.local_mem_bytes / 1024.0);
  printf("  element types            %u\n", caps.dtype_count);
  printf("  per-instruction limits:\n");
  printf("    matrix engine (MMA/MATVEC)   %ux%ux%u\n", caps.mma_dim,
         caps.mma_dim, caps.mma_dim);
  printf("    vector ALU (EW/FMA/CONVERT)  %u elements\n", caps.vector_lanes);
  printf("    special function unit (ACT)  %u elements\n", caps.sfu_lanes);
  printf("    row unit (REDUCE/SOFTMAX/NORM) 1 row, width <= %u\n",
         caps.max_row_width);
  printf("    transpose / gather / im2col  %u rows\n", caps.transpose_dim);
  printf("    DMA descriptor               %.0f KB\n",
         (double)caps.dma_max_bytes / 1024.0);

  // Anything larger has to be tiled by whoever issues the instructions. This is
  // the number of instructions this GEMM will therefore take:
  const uint32_t tile = caps.mma_dim;
  uint32_t m_tiles = (M + tile - 1) / tile;
  uint32_t n_tiles = (N + tile - 1) / tile;
  uint32_t k_tiles = (K + tile - 1) / tile;
  printf(
      "\nGEMM %dx%dx%d -> %u x %u output tiles x %u k-tiles"
      " = %u MMA + %u DMA instructions\n",
      M, N, K, m_tiles, n_tiles, k_tiles, m_tiles * n_tiles * k_tiles,
      m_tiles * n_tiles * (2 * k_tiles + 1));

  //=== Host data + reference ==============================================
  float* a = malloc(sizeof(float) * M * K);
  float* b = malloc(sizeof(float) * K * N);
  float* c = malloc(sizeof(float) * M * N);
  double* ref = malloc(sizeof(double) * M * N);
  fill(a, (size_t)M * K, 1);
  fill(b, (size_t)K * N, 2);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        acc += (double)a[i * K + k] * b[k * N + j];
      }
      ref[i * N + j] = acc;
    }
  }

  const emunpu_dtype_t f32 = EMUNPU_DT_FP32;
  uint32_t n_a = M * K, n_b = K * N, n_c = M * N;
  emunpu_tensor_id_t da, db, dc;
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_a, "A", &da));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_b, "B", &db));
  CHECK_OK(emunpu_device_alloc(dev, f32, 1, &n_c, "C", &dc));
  CHECK_OK(emunpu_device_upload(dev, da, a, sizeof(float) * n_a));
  CHECK_OK(emunpu_device_upload(dev, db, b, sizeof(float) * n_b));

  //=== 2. A hand-written tiled GEMM =======================================
  // Operand addresses in the global address space -- the same thing a compiler
  // would bake into a schedule.
  emunpu_global_mem_t* gm = emunpu_device_global_mem(dev);
  uint64_t a_base = 0, b_base = 0, c_base = 0;
  CHECK_OK(emunpu_global_mem_offset(gm, da, &a_base));
  CHECK_OK(emunpu_global_mem_offset(gm, db, &b_base));
  CHECK_OK(emunpu_global_mem_offset(gm, dc, &c_base));

  // One A/B/C tile buffer per core, reused by every tile that core handles (a
  // core's instructions execute in order, so the reuse is safe).
  const uint64_t tile_bytes = (uint64_t)tile * tile * sizeof(float);
  uint64_t a_local[IREE_EMUNPU_NUM_CORES], b_local[IREE_EMUNPU_NUM_CORES];
  uint64_t c_local[IREE_EMUNPU_NUM_CORES];
  for (uint32_t core = 0; core < caps.num_cores; ++core) {
    CHECK_OK(emunpu_device_local_reset(dev, core));
    CHECK_OK(emunpu_device_local_alloc(dev, core, tile_bytes, &a_local[core]));
    CHECK_OK(emunpu_device_local_alloc(dev, core, tile_bytes, &b_local[core]));
    CHECK_OK(emunpu_device_local_alloc(dev, core, tile_bytes, &c_local[core]));
  }

  uint32_t tile_index = 0;
  for (uint32_t m0 = 0; m0 < M; m0 += tile) {
    uint32_t tm = (M - m0 < tile) ? (M - m0) : tile;
    for (uint32_t n0 = 0; n0 < N; n0 += tile, ++tile_index) {
      uint32_t tn = (N - n0 < tile) ? (N - n0) : tile;
      // Our own core assignment: round-robin over output tiles.
      uint32_t core = tile_index % caps.num_cores;
      for (uint32_t k0 = 0; k0 < K; k0 += tile) {
        uint32_t tk = (K - k0 < tile) ? (K - k0) : tile;

        // A[m0:+tm, k0:+tk] : global -> local (strided read, packed write).
        emunpu_instr_t dma_a = {0};
        dma_a.op = EMUNPU_I_DMA_G2L;
        dma_a.a_dtype = dma_a.c_dtype = f32;
        dma_a.M = tm;
        dma_a.N = tk;
        dma_a.a_offset = a_base + ((uint64_t)m0 * K + k0) * sizeof(float);
        dma_a.a_row_stride = K * sizeof(float);
        dma_a.c_offset = a_local[core];
        CHECK_OK(emunpu_device_enqueue(dev, core, &dma_a));

        // B[k0:+tk, n0:+tn] : global -> local.
        emunpu_instr_t dma_b = dma_a;
        dma_b.M = tk;
        dma_b.N = tn;
        dma_b.a_offset = b_base + ((uint64_t)k0 * N + n0) * sizeof(float);
        dma_b.a_row_stride = N * sizeof(float);
        dma_b.c_offset = b_local[core];
        CHECK_OK(emunpu_device_enqueue(dev, core, &dma_b));

        // C_tile (+)= A_tile * B_tile, entirely within the scratchpad.
        emunpu_instr_t mma = {0};
        mma.op = EMUNPU_I_MMA16;
        mma.a_dtype = mma.b_dtype = mma.acc_dtype = mma.c_dtype = f32;
        mma.M = tm;
        mma.N = tn;
        mma.K = tk;
        mma.a_space = mma.b_space = mma.c_space = EMUNPU_SPACE_LOCAL;
        mma.a_offset = a_local[core];
        mma.b_offset = b_local[core];
        mma.c_offset = c_local[core];
        mma.flags = k0 ? EMUNPU_MMA_ACCUMULATE : 0;
        CHECK_OK(emunpu_device_enqueue(dev, core, &mma));
      }
      // The finished tile : local -> global.
      emunpu_instr_t dma_c = {0};
      dma_c.op = EMUNPU_I_DMA_L2G;
      dma_c.a_dtype = dma_c.c_dtype = f32;
      dma_c.M = tm;
      dma_c.N = tn;
      dma_c.a_offset = c_local[core];
      dma_c.c_offset = c_base + ((uint64_t)m0 * N + n0) * sizeof(float);
      dma_c.c_row_stride = N * sizeof(float);
      CHECK_OK(emunpu_device_enqueue(dev, core, &dma_c));
    }
  }

  // An instruction that exceeds a unit's width is refused, not slowly executed:
  emunpu_instr_t too_big = {0};
  too_big.op = EMUNPU_I_MMA16;
  too_big.a_dtype = too_big.b_dtype = too_big.c_dtype = f32;
  too_big.M = too_big.N = too_big.K = tile + 1;
  printf("\nenqueueing a %ux%ux%u MMA (one past the engine's %ux%u tile): %s\n",
         tile + 1, tile + 1, tile + 1, tile, tile,
         emunpu_status_name(emunpu_device_enqueue(dev, 0, &too_big)));

  CHECK_OK(emunpu_device_launch(dev));
  CHECK_OK(emunpu_device_download(dev, dc, c, sizeof(float) * n_c));
  double err_raw = max_abs_err(c, ref, (size_t)M * N);

  printf("\nhand-scheduled GEMM: max abs error %.3e  -> %s\n", err_raw,
         err_raw < 1e-4 ? "PASS" : "FAIL");
  printf("  per-core work (instructions / MMA issues / DMA bytes):\n");
  for (uint32_t core = 0; core < caps.num_cores; ++core) {
    emunpu_core_stats_t st;
    CHECK_OK(emunpu_device_core_stats(dev, core, &st));
    printf(
        "    core%-2u %6llu instr  %5llu mma  %8llu B  (%.0f KB scratchpad"
        " in use)\n",
        core, (unsigned long long)st.instr_count,
        (unsigned long long)st.mma_count, (unsigned long long)st.dma_bytes,
        (double)st.local_used / 1024.0);
  }

  //=== 3. Choosing how many cores an operation uses =======================
  printf("\nsame GEMM through the high-level API, per execution scope:\n");
  int scope_ok = 1;
  for (uint32_t use = 1; use <= caps.num_cores; use *= 2) {
    emunpu_core_stats_t before[IREE_EMUNPU_NUM_CORES];
    for (uint32_t i = 0; i < caps.num_cores; ++i) {
      CHECK_OK(emunpu_device_core_stats(dev, i, &before[i]));
    }
    CHECK_OK(emunpu_device_set_exec_scope(dev, 0, use));
    CHECK_OK(emunpu_matmul(dev, NULL, dc, da, db, M, N, K, /*transpose_b=*/0));
    CHECK_OK(emunpu_device_download(dev, dc, c, sizeof(float) * n_c));
    double err = max_abs_err(c, ref, (size_t)M * N);
    if (err >= 1e-4) scope_ok = 0;

    printf("  %2u core%s: err %.3e  instructions per core [", use,
           use == 1 ? " " : "s", err);
    for (uint32_t i = 0; i < caps.num_cores; ++i) {
      emunpu_core_stats_t after;
      CHECK_OK(emunpu_device_core_stats(dev, i, &after));
      printf("%s%llu", i ? " " : "",
             (unsigned long long)(after.instr_count - before[i].instr_count));
    }
    printf("]\n");
  }
  emunpu_device_reset_exec_scope(dev);
  printf("-> %s\n", (err_raw < 1e-4 && scope_ok) ? "PASS" : "FAIL");

  free(a);
  free(b);
  free(c);
  free(ref);
  emunpu_device_free(dev, da);
  emunpu_device_free(dev, db);
  emunpu_device_free(dev, dc);
  emunpu_device_destroy(dev);
  return (err_raw < 1e-4 && scope_ok) ? 0 : 1;
}
