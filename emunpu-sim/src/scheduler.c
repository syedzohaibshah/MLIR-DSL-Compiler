// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emunpu/scheduler.h"

#include <string.h>

#define TILE IREE_EMUNPU_MMA_DIM

static uint32_t emunpu_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Per-core fixed local scratchpad layout reused across that core's tiles: one
// 16x16 tile each for A, B, and the C accumulator. The C tile is DMA'd out
// before the next tile reuses the space, so in-order execution is safe.
typedef struct emunpu_core_tiles_t {
  uint64_t a_local;
  uint64_t b_local;
  uint64_t c_local;
} emunpu_core_tiles_t;

// Resets each core's queue + local scratchpad and reserves its fixed A/B/C
// 16x16 local tiles (filling |local|). Clamps *|num_cores_inout| to the number
// of physical cores. Both scheduling entry points share this prologue.
static emunpu_status_t emunpu_setup_cores_local(emunpu_core_t* cores,
                                                uint32_t* num_cores_inout,
                                                size_t as, size_t bs, size_t cs,
                                                emunpu_core_tiles_t* local) {
  uint32_t num_cores = *num_cores_inout;
  if (num_cores > IREE_EMUNPU_NUM_CORES) num_cores = IREE_EMUNPU_NUM_CORES;
  for (uint32_t c = 0; c < num_cores; ++c) {
    emunpu_core_queue_reset(&cores[c]);
    emunpu_core_local_reset(&cores[c]);
    emunpu_status_t st;
    if ((st = emunpu_core_local_alloc(&cores[c], (uint64_t)TILE * TILE * as,
                                      &local[c].a_local)) != EMUNPU_OK ||
        (st = emunpu_core_local_alloc(&cores[c], (uint64_t)TILE * TILE * bs,
                                      &local[c].b_local)) != EMUNPU_OK ||
        (st = emunpu_core_local_alloc(&cores[c], (uint64_t)TILE * TILE * cs,
                                      &local[c].c_local)) != EMUNPU_OK) {
      return st;
    }
  }
  *num_cores_inout = num_cores;
  return EMUNPU_OK;
}

// Emits the DMA + MMA16 (+ optional C writeback) instruction stream for one
// 16x16 (m0,n0,k0) tile onto |core|, using the core's fixed local A/B/C tiles
// |lt|. |accumulate| selects MMA accumulate vs. overwrite (2nd..nth K-tile);
// |writeback| DMAs the finished C tile back to global after the MMA (last
// K-tile). Shared by emunpu_schedule_gemm (decides the tiling) and
// emunpu_run_gemm_schedule (replays a compiler-decided schedule).
static emunpu_status_t emunpu_emit_gemm_tile(
    emunpu_core_t* core, const emunpu_gemm_desc_t* gemm,
    const emunpu_core_tiles_t* lt, uint32_t m0, uint32_t n0, uint32_t k0,
    uint32_t tm, uint32_t tn, uint32_t tkk, int accumulate, int writeback) {
  const size_t as = emunpu_dtype_size(gemm->a_dtype);
  const size_t bs = emunpu_dtype_size(gemm->b_dtype);
  const size_t cs = emunpu_dtype_size(gemm->c_dtype);
  const uint32_t N = gemm->N, K = gemm->K;
  emunpu_status_t st;

  // DMA the A subtile A[m0:m0+tm, k0:k0+tkk] (global, row stride K) into the
  // core's local A tile (contiguous, row stride tkk).
  emunpu_instr_t dma_a = {0};
  dma_a.op = EMUNPU_I_DMA_G2L;
  dma_a.c_dtype = gemm->a_dtype;  // DMA element type
  dma_a.M = tm;
  dma_a.N = tkk;
  dma_a.a_space = EMUNPU_SPACE_GLOBAL;
  dma_a.a_offset = gemm->a_offset + (uint64_t)(m0 * K + k0) * as;
  dma_a.a_row_stride = (uint32_t)((uint64_t)K * as);
  dma_a.c_space = EMUNPU_SPACE_LOCAL;
  dma_a.c_offset = lt->a_local;
  dma_a.c_row_stride = (uint32_t)((uint64_t)tkk * as);
  if ((st = emunpu_core_enqueue(core, &dma_a)) != EMUNPU_OK) return st;

  // DMA the B subtile into the local B tile. Layout depends on transpose.
  emunpu_instr_t dma_b = {0};
  dma_b.op = EMUNPU_I_DMA_G2L;
  dma_b.c_dtype = gemm->b_dtype;
  dma_b.a_space = EMUNPU_SPACE_GLOBAL;
  dma_b.c_space = EMUNPU_SPACE_LOCAL;
  dma_b.c_offset = lt->b_local;
  if (gemm->transpose_b) {
    // B is N x K; subtile B[n0:n0+tn, k0:k0+tkk].
    dma_b.M = tn;
    dma_b.N = tkk;
    dma_b.a_offset = gemm->b_offset + (uint64_t)(n0 * K + k0) * bs;
    dma_b.a_row_stride = (uint32_t)((uint64_t)K * bs);
    dma_b.c_row_stride = (uint32_t)((uint64_t)tkk * bs);
  } else {
    // B is K x N; subtile B[k0:k0+tkk, n0:n0+tn].
    dma_b.M = tkk;
    dma_b.N = tn;
    dma_b.a_offset = gemm->b_offset + (uint64_t)(k0 * N + n0) * bs;
    dma_b.a_row_stride = (uint32_t)((uint64_t)N * bs);
    dma_b.c_row_stride = (uint32_t)((uint64_t)tn * bs);
  }
  if ((st = emunpu_core_enqueue(core, &dma_b)) != EMUNPU_OK) return st;

  // MMA16: local C tile (+)= local A tile * local B tile. Overwrite on the
  // first K-tile, accumulate after.
  emunpu_instr_t mma = {0};
  mma.op = EMUNPU_I_MMA16;
  mma.a_dtype = gemm->a_dtype;
  mma.b_dtype = gemm->b_dtype;
  mma.acc_dtype = gemm->c_dtype;
  mma.c_dtype = gemm->c_dtype;
  mma.M = tm;
  mma.N = tn;
  mma.K = tkk;
  mma.a_space = EMUNPU_SPACE_LOCAL;
  mma.a_offset = lt->a_local;
  mma.a_row_stride = (uint32_t)((uint64_t)tkk * as);
  mma.b_space = EMUNPU_SPACE_LOCAL;
  mma.b_offset = lt->b_local;
  mma.b_row_stride = gemm->transpose_b ? (uint32_t)((uint64_t)tkk * bs)
                                       : (uint32_t)((uint64_t)tn * bs);
  mma.c_space = EMUNPU_SPACE_LOCAL;
  mma.c_offset = lt->c_local;
  mma.c_row_stride = (uint32_t)((uint64_t)tn * cs);
  mma.flags = (gemm->transpose_b ? EMUNPU_MMA_TRANSPOSE_B : 0) |
              (accumulate ? EMUNPU_MMA_ACCUMULATE : 0);
  if ((st = emunpu_core_enqueue(core, &mma)) != EMUNPU_OK) return st;

  if (!writeback) return EMUNPU_OK;

  // DMA the finished C tile back to global memory C[m0:m0+tm, n0:n0+tn].
  emunpu_instr_t dma_c = {0};
  dma_c.op = EMUNPU_I_DMA_L2G;
  dma_c.c_dtype = gemm->c_dtype;
  dma_c.M = tm;
  dma_c.N = tn;
  dma_c.a_space = EMUNPU_SPACE_LOCAL;
  dma_c.a_offset = lt->c_local;
  dma_c.a_row_stride = (uint32_t)((uint64_t)tn * cs);
  dma_c.c_space = EMUNPU_SPACE_GLOBAL;
  dma_c.c_offset = gemm->c_offset + (uint64_t)(m0 * N + n0) * cs;
  dma_c.c_row_stride = (uint32_t)((uint64_t)N * cs);
  return emunpu_core_enqueue(core, &dma_c);
}

emunpu_status_t emunpu_schedule_gemm(emunpu_core_t* cores, uint32_t num_cores,
                                     const emunpu_gemm_desc_t* gemm,
                                     uint32_t* out_tile_count) {
  if (!cores || !gemm || num_cores == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  size_t as = emunpu_dtype_size(gemm->a_dtype);
  size_t bs = emunpu_dtype_size(gemm->b_dtype);
  size_t cs = emunpu_dtype_size(gemm->c_dtype);
  if (!as || !bs || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;

  emunpu_core_tiles_t local[IREE_EMUNPU_NUM_CORES];
  emunpu_status_t st =
      emunpu_setup_cores_local(cores, &num_cores, as, bs, cs, local);
  if (st != EMUNPU_OK) return st;

  const uint32_t M = gemm->M, N = gemm->N, K = gemm->K;
  const uint32_t row_tiles = (M + TILE - 1) / TILE;
  const uint32_t col_tiles = (N + TILE - 1) / TILE;
  const uint32_t k_tiles = (K + TILE - 1) / TILE;
  uint32_t tile_index = 0;

  for (uint32_t ti = 0; ti < row_tiles; ++ti) {
    uint32_t m0 = ti * TILE;
    uint32_t tm = emunpu_min_u32(TILE, M - m0);
    for (uint32_t tj = 0; tj < col_tiles; ++tj, ++tile_index) {
      uint32_t n0 = tj * TILE;
      uint32_t tn = emunpu_min_u32(TILE, N - n0);
      uint32_t core_id = tile_index % num_cores;
      emunpu_core_t* core = &cores[core_id];
      const emunpu_core_tiles_t* lt = &local[core_id];

      // Accumulate this output tile over all K-tiles.
      for (uint32_t tk = 0; tk < k_tiles; ++tk) {
        uint32_t k0 = tk * TILE;
        uint32_t tkk = emunpu_min_u32(TILE, K - k0);
        st = emunpu_emit_gemm_tile(core, gemm, lt, m0, n0, k0, tm, tn, tkk,
                                   /*accumulate=*/tk > 0,
                                   /*writeback=*/tk == k_tiles - 1);
        if (st != EMUNPU_OK) return st;
      }
    }
  }

  if (out_tile_count) *out_tile_count = tile_index;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_run_gemm_schedule(emunpu_core_t* cores,
                                         uint32_t num_cores,
                                         const emunpu_gemm_desc_t* gemm,
                                         const emunpu_gemm_tile_t* tiles,
                                         uint32_t tile_count) {
  if (!cores || !gemm || !tiles || num_cores == 0) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  size_t as = emunpu_dtype_size(gemm->a_dtype);
  size_t bs = emunpu_dtype_size(gemm->b_dtype);
  size_t cs = emunpu_dtype_size(gemm->c_dtype);
  if (!as || !bs || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;

  emunpu_core_tiles_t local[IREE_EMUNPU_NUM_CORES];
  emunpu_status_t st =
      emunpu_setup_cores_local(cores, &num_cores, as, bs, cs, local);
  if (st != EMUNPU_OK) return st;

  // Replay the compiler's schedule verbatim -- no tiling decision happens here.
  // The compiler always bakes a fixed IREE_EMUNPU_NUM_CORES-way round-robin, so
  // when fewer cores are online (IREE_EMUNPU_NUM_CORES env) we fold each tile's
  // core id onto an online core with the same modulo. This preserves the
  // property that all k-tiles of one output tile (which share a core id) stay
  // on the same core and execute contiguously, so accumulation stays correct.
  for (uint32_t i = 0; i < tile_count; ++i) {
    const emunpu_gemm_tile_t* t = &tiles[i];
    uint32_t cid = t->core_id % num_cores;
    st = emunpu_emit_gemm_tile(&cores[cid], gemm, &local[cid], t->m0, t->n0,
                               t->k0, t->tm, t->tn, t->tk, t->accumulate,
                               t->writeback);
    if (st != EMUNPU_OK) return st;
  }
  return EMUNPU_OK;
}
