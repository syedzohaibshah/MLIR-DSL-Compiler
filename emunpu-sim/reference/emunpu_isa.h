// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Emulated NPU "instruction set" / executable blob contract.
//
// This header is the single source of truth for the binary format produced by
// the `emunpu` compiler target backend and consumed by the `emunpu` HAL driver.
// It is intentionally a small set of plain-C POD structs (no flatbuffers): the
// emulated NPU is coarse-grained and understands a handful of whole-tensor ops
// (matmul/add/relu/copy) rather than scalar code, so the "program" for a single
// dispatch is one (or a few) high-level command records referencing dispatch
// bindings by ordinal.
//
// It is included by BOTH the compiler plugin (C++) and the runtime driver (C),
// so keep it dependency-free and C/C++ compatible.
//
// Blob layout (all offsets are byte offsets from the start of the blob):
//   [emunpu_header_t]
//   [emunpu_export_t * header.export_count]
//   [emunpu_command_t * header.command_count]
//   [emunpu_matmul_tile_t * header.tile_count]
//   [string table bytes]  (export names, not NUL-terminated)
//
// Tiling note: the compiler tiles each matmul into MMA_DIM x MMA_DIM blocks and
// assigns each block to a core AT COMPILE TIME, emitting the resulting schedule
// as the tile records a MATMUL command points at (first_tile/tile_count). This
// mirrors how the CUDA/ROCm backends bake the workgroup grid into the HAL
// executable: the runtime no longer *decides* the tiling, it just replays the
// schedule (computing per-tile addresses from the tile coordinates, the way a
// GPU kernel computes per-block addresses from blockIdx). IREE_EMUNPU_MMA_DIM
// and IREE_EMUNPU_NUM_CORES (emunpu_arch.h) are the physical device parameters
// the compiler tiles against, so both halves must agree on them.
//===----------------------------------------------------------------------===//

#ifndef IREE_HAL_DRIVERS_EMUNPU_EMUNPU_ISA_H_
#define IREE_HAL_DRIVERS_EMUNPU_EMUNPU_ISA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// HAL executable format string. Must match the format the compiler stamps on
// the `hal.executable.binary` op and the format the runtime loader accepts.
#define IREE_HAL_EMUNPU_EXECUTABLE_FORMAT "emunpu-cmd-fb"

// 'E' 'N' 'P' 'U' little-endian.
#define IREE_HAL_EMUNPU_MAGIC 0x55504E45u
// v2 adds: multi-command (fused) dispatches, scratch-slot operand/result refs
// for intermediate tensors, and a broadcast period for the elementwise add.
// v4 adds: a compile-time matmul tile schedule (emunpu_matmul_tile_t records +
// header.tile_count); MATMUL commands carry first_tile/tile_count into it.
#define IREE_HAL_EMUNPU_VERSION 4u

// Coarse op the emulated NPU compute engine knows how to execute.
enum iree_hal_emunpu_opcode_e {
  IREE_HAL_EMUNPU_OP_MATMUL =
      1,                        // result[M,N] = lhs[M,K] * rhs[K,N] (+= result)
  IREE_HAL_EMUNPU_OP_ADD = 2,   // result = lhs + rhs (elementwise)
  IREE_HAL_EMUNPU_OP_RELU = 3,  // result = max(operand, 0) (elementwise)
  IREE_HAL_EMUNPU_OP_COPY = 4,  // result = operand (elementwise copy)
};
typedef uint32_t iree_hal_emunpu_opcode_t;

// Element type the command operates on.
enum iree_hal_emunpu_dtype_e {
  IREE_HAL_EMUNPU_DTYPE_F32 = 1,
  IREE_HAL_EMUNPU_DTYPE_I32 = 2,
};
typedef uint32_t iree_hal_emunpu_dtype_t;

// MATMUL layout flags (stored in command dims[3]). The lhs is always A:MxK and
// the output C:MxN; only the rhs layout varies.
enum iree_hal_emunpu_matmul_flags_e {
  IREE_HAL_EMUNPU_MATMUL_STD = 0,          // B is K x N: C[m,n] += A[m,k]*B[k,n]
  IREE_HAL_EMUNPU_MATMUL_TRANSPOSE_B = 1,  // B is N x K: C[m,n] += A[m,k]*B[n,k]
};

// Per-tile flags (emunpu_matmul_tile_t.flags). A matmul is tiled into
// MMA_DIM x MMA_DIM output blocks, each accumulated over the K dimension in
// MMA_DIM-wide k-tiles. ACCUMULATE marks the 2nd..nth k-tile of an output block
// (the MMA adds into the running C tile instead of overwriting); WRITEBACK marks
// the last k-tile, after which the finished C block is DMA'd back to global.
enum iree_hal_emunpu_tile_flags_e {
  IREE_HAL_EMUNPU_TILE_ACCUMULATE = 1,
  IREE_HAL_EMUNPU_TILE_WRITEBACK = 2,
};

// One MMA_DIM x MMA_DIM (x MMA_DIM in K) matmul tile produced by the compiler.
// Coordinates are element offsets into the matmul's M/N/K iteration space and
// extents are clamped at the matrix edges (tm/tn/tk <= MMA_DIM). The record
// carries no absolute memory offset: those depend on runtime buffer staging, so
// the runtime derives them from these shape-relative coordinates plus the
// operand base (matching the legacy runtime scheduler's arithmetic).
typedef struct iree_hal_emunpu_matmul_tile_t {
  uint32_t core_id;  // core this tile is scheduled on (< IREE_EMUNPU_NUM_CORES)
  uint32_t m0, n0, k0;  // tile origin in the M/N/K iteration space
  uint32_t tm, tn, tk;  // tile extents (<= IREE_EMUNPU_MMA_DIM)
  uint32_t flags;       // bitmask of iree_hal_emunpu_tile_flags_e
} iree_hal_emunpu_matmul_tile_t;

// Operand/result reference encoding (used by operand_bindings[] / result_binding):
//   >=  0 : dispatch binding ordinal (index into binding_ptrs).
//   == -1 : unused slot.
//   <= -2 : intermediate "scratch" slot s = (-ref) - 2 (so -2 => slot 0). The
//           runtime allocates one buffer per scratch slot for the lifetime of a
//           single dispatch so a fused op chain (e.g. matmul -> add -> relu) can
//           pass results between commands without a HAL binding.
#define IREE_HAL_EMUNPU_REF_UNUSED (-1)
#define IREE_HAL_EMUNPU_REF_IS_SCRATCH(ref) ((ref) <= -2)
#define IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(ref) ((uint32_t)((-(ref)) - 2))
#define IREE_HAL_EMUNPU_SCRATCH_REF(slot) (-((int32_t)(slot)) - 2)

// A single coarse NPU command. Operands/results reference dispatch bindings or
// scratch slots per the encoding above. Per opcode:
//   MATMUL: dims={M,N,K,flags};        operands={lhs,rhs,-1}; result=out
//           (flags: iree_hal_emunpu_matmul_flags_e selecting the rhs layout)
//           tiles[first_tile .. first_tile+tile_count) is the compile-time tile
//           schedule the runtime replays (see emunpu_matmul_tile_t).
//   ADD:    dims={elem_count,period};  operands={lhs,rhs,-1}; result=out
//           (period 0 => equal-shape; period>0 => rhs is broadcast and indexed
//            as rhs[i % period], used for nn.Linear-style row bias)
//   RELU:   dims={elem_count,...};     operands={in,-1,-1};   result=out
//   COPY:   dims={elem_count,...};     operands={in,-1,-1};   result=out
// first_tile/tile_count are only meaningful for MATMUL (0 for elementwise ops,
// which the runtime splits by element range with no compile-time schedule).
typedef struct iree_hal_emunpu_command_t {
  iree_hal_emunpu_opcode_t opcode;
  iree_hal_emunpu_dtype_t dtype;
  uint32_t dims[4];
  uint32_t first_tile;   // MATMUL: index into the tiles array
  uint32_t tile_count;   // MATMUL: number of tile records for this command
  int32_t operand_bindings[3];
  int32_t result_binding;
  // Byte offsets added to each operand/result binding pointer. IREE can pack
  // several tensors into one buffer and address them via a static subspan
  // offset (e.g. two transposed weights in a single constant buffer), so a
  // binding ordinal alone is not enough -- the offset selects the sub-region.
  // Ignored for scratch refs (which are always whole-buffer).
  uint32_t operand_offsets[3];
  uint32_t result_offset;
} iree_hal_emunpu_command_t;

// Per-export record. Commands for an export are the contiguous range
// commands[first_command .. first_command + command_count).
typedef struct iree_hal_emunpu_export_t {
  uint32_t name_offset;     // byte offset into the string table
  uint32_t name_length;     // length in bytes (name is not NUL-terminated)
  uint32_t first_command;   // index into the commands array
  uint32_t command_count;   // number of commands for this export
  uint32_t binding_count;   // distinct bindings the dispatch consumes
  uint32_t constant_count;  // 32-bit constants the dispatch consumes
} iree_hal_emunpu_export_t;

typedef struct iree_hal_emunpu_header_t {
  uint32_t magic;          // IREE_HAL_EMUNPU_MAGIC
  uint32_t version;        // IREE_HAL_EMUNPU_VERSION
  uint32_t export_count;   // number of emunpu_export_t records
  uint32_t command_count;  // total number of emunpu_command_t records
  uint32_t tile_count;     // total number of emunpu_matmul_tile_t records
  uint32_t string_table_offset;
  uint32_t string_table_size;
} iree_hal_emunpu_header_t;

// Offset (in bytes from blob start) of the export table.
static inline uint32_t iree_hal_emunpu_export_table_offset(void) {
  return (uint32_t)sizeof(iree_hal_emunpu_header_t);
}

// Offset (in bytes from blob start) of the command array.
static inline uint32_t iree_hal_emunpu_command_table_offset(
    uint32_t export_count) {
  return (uint32_t)sizeof(iree_hal_emunpu_header_t) +
         export_count * (uint32_t)sizeof(iree_hal_emunpu_export_t);
}

// Offset (in bytes from blob start) of the tile array (immediately after the
// command array).
static inline uint32_t iree_hal_emunpu_tile_table_offset(
    uint32_t export_count, uint32_t command_count) {
  return iree_hal_emunpu_command_table_offset(export_count) +
         command_count * (uint32_t)sizeof(iree_hal_emunpu_command_t);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_EMUNPU_EMUNPU_ISA_H_
