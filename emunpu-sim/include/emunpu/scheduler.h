// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Workload scheduler / tiler.
//
// Partitions a coarse operation (today: a GEMM) into 16x16 tiles and fans the
// resulting DMA + MMA16 instruction streams across the available cores. For each
// 16x16 output tile it generates, per assigned core, a stream of
//   DMA A-subtile g->l ; DMA B-subtile g->l ; MMA16 acc ; (repeat over K-tiles)
//   DMA C-tile l->g
// Output tiles are distributed round-robin so the cores share the work; ragged
// edges (dims not multiples of 16) are handled with partial tiles.
//
// Two entry points share the same per-tile instruction emission:
//   - emunpu_schedule_gemm     : the device-native API path *decides* the tiling
//                                here (used by the standalone CUDA-style API and
//                                its unit tests).
//   - emunpu_run_gemm_schedule : the IREE HAL path *replays* a tile schedule the
//                                compiler already decided (see emunpu_isa.h's
//                                emunpu_matmul_tile_t). The runtime no longer
//                                tiles for compiled programs.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_SCHEDULER_H_
#define EMUNPU_SCHEDULER_H_

#include "emunpu/core.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A GEMM expressed in global-memory byte offsets: C[M x N] = A[M x K] * B,
// where B is K x N (transpose_b == 0) or N x K (transpose_b == 1).
typedef struct emunpu_gemm_desc_t {
  uint64_t a_offset;
  uint64_t b_offset;
  uint64_t c_offset;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  emunpu_dtype_t a_dtype;
  emunpu_dtype_t b_dtype;
  emunpu_dtype_t c_dtype;  // also the MMA accumulator/output type
  int transpose_b;
} emunpu_gemm_desc_t;

// Tiles |gemm| into 16x16 blocks and enqueues the DMA/MMA16 instruction streams
// onto |cores| (round-robin over output tiles). Resets each core's queue and
// local scratchpad first. Does not execute -- the caller runs the cores.
// Returns the number of output tiles scheduled in |out_tile_count| (optional).
EMUNPU_API emunpu_status_t emunpu_schedule_gemm(emunpu_core_t* cores, uint32_t num_cores,
                                     const emunpu_gemm_desc_t* gemm,
                                     uint32_t* out_tile_count);

// A single precomputed matmul tile (mirror of the HAL ISA's
// iree_hal_emunpu_matmul_tile_t, kept IREE-independent here). Coordinates are
// element offsets into the M/N/K space; extents are clamped at the edges.
typedef struct emunpu_gemm_tile_t {
  uint32_t core_id;
  uint32_t m0, n0, k0;
  uint32_t tm, tn, tk;
  int accumulate;  // MMA accumulates into C instead of overwriting (k0 > 0)
  int writeback;   // DMA the finished C tile to global after this MMA (last k)
} emunpu_gemm_tile_t;

// Replays a compiler-produced tile schedule for |gemm| onto |cores|: enqueues
// the DMA/MMA16 instructions for each tile onto tile.core_id, without deciding
// the tiling itself. Resets each core's queue/scratchpad first. Does not
// execute -- the caller runs the cores. The tiles must be ordered as the
// compiler emits them (per output tile, k ascending), which the fixed per-core
// A/B/C scratchpad reuse relies on.
EMUNPU_API emunpu_status_t emunpu_run_gemm_schedule(emunpu_core_t* cores,
                                         uint32_t num_cores,
                                         const emunpu_gemm_desc_t* gemm,
                                         const emunpu_gemm_tile_t* tiles,
                                         uint32_t tile_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_SCHEDULER_H_
