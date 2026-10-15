// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/emunpu/emunpu_hw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/hal/drivers/emunpu/emunpu_device.h"
#include "iree/hal/drivers/emunpu/emunpu_viz.h"

//===----------------------------------------------------------------------===//
// Execution trace
//
// Optional, opt-in proof that a workload actually ran on the emulated NPU
// compute engine (rather than e.g. falling back to another backend). Set the
// environment variable IREE_EMUNPU_TRACE=1 and every coarse op the engine
// executes is logged to stderr with a running op counter, e.g.:
//
//   [emunpu-hw] #1 MATMUL f32 M=4 N=16 K=8 transpose_b=1 in=[b0,b1] out=s0
//   [emunpu-hw] #2 ADD    f32 n=64 period=16 in=[s0,b2] out=s1
//   [emunpu-hw] #3 RELU   f32 n=64 in=[s1] out=b3
//===----------------------------------------------------------------------===//

// 0 = unknown (not yet read), 1 = enabled, 2 = disabled.
static int iree_hal_emunpu_trace_state = 0;
static uint64_t iree_hal_emunpu_op_counter = 0;

bool iree_hal_emunpu_hw_trace_enabled(void) {
  if (iree_hal_emunpu_trace_state == 0) {
    const char* v = getenv("IREE_EMUNPU_TRACE");
    iree_hal_emunpu_trace_state = (v && v[0] != '\0' && v[0] != '0') ? 1 : 2;
  }
  return iree_hal_emunpu_trace_state == 1;
}

static const char* iree_hal_emunpu_opcode_name(
    iree_hal_emunpu_opcode_t opcode) {
  switch (opcode) {
    case IREE_HAL_EMUNPU_OP_MATMUL:
      return "MATMUL";
    case IREE_HAL_EMUNPU_OP_ADD:
      return "ADD";
    case IREE_HAL_EMUNPU_OP_RELU:
      return "RELU";
    case IREE_HAL_EMUNPU_OP_COPY:
      return "COPY";
    default:
      return "????";
  }
}

// Formats an operand/result ref the way the ISA encodes it: bindings as bN,
// scratch slots as sN, and unused slots as '-'.
static void iree_hal_emunpu_format_ref(int32_t ref, char* out,
                                       size_t out_size) {
  if (ref == IREE_HAL_EMUNPU_REF_UNUSED) {
    snprintf(out, out_size, "-");
  } else if (IREE_HAL_EMUNPU_REF_IS_SCRATCH(ref)) {
    snprintf(out, out_size, "s%u", IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(ref));
  } else {
    snprintf(out, out_size, "b%d", ref);
  }
}

void iree_hal_emunpu_hw_trace_command(
    const iree_hal_emunpu_command_t* command) {
  const char* dt = command->dtype == IREE_HAL_EMUNPU_DTYPE_F32 ? "f32" : "i32";
  char in0[8], in1[8], in2[8], out[8];
  iree_hal_emunpu_format_ref(command->operand_bindings[0], in0, sizeof(in0));
  iree_hal_emunpu_format_ref(command->operand_bindings[1], in1, sizeof(in1));
  iree_hal_emunpu_format_ref(command->operand_bindings[2], in2, sizeof(in2));
  iree_hal_emunpu_format_ref(command->result_binding, out, sizeof(out));
  uint64_t n = ++iree_hal_emunpu_op_counter;
  switch (command->opcode) {
    case IREE_HAL_EMUNPU_OP_MATMUL:
      fprintf(stderr,
              "[emunpu-hw] #%llu MATMUL %s M=%u N=%u K=%u transpose_b=%u "
              "in=[%s,%s] out=%s\n",
              (unsigned long long)n, dt, command->dims[0], command->dims[1],
              command->dims[2],
              (command->dims[3] & IREE_HAL_EMUNPU_MATMUL_TRANSPOSE_B) ? 1u : 0u,
              in0, in1, out);
      break;
    case IREE_HAL_EMUNPU_OP_ADD:
      fprintf(stderr,
              "[emunpu-hw] #%llu ADD    %s n=%u period=%u in=[%s,%s] out=%s\n",
              (unsigned long long)n, dt, command->dims[0], command->dims[1],
              in0, in1, out);
      break;
    default:  // RELU / COPY (and any unknown).
      fprintf(stderr, "[emunpu-hw] #%llu %-6s %s n=%u in=[%s] out=%s\n",
              (unsigned long long)n,
              iree_hal_emunpu_opcode_name(command->opcode), dt,
              command->dims[0], in0, out);
      break;
  }
}

//===----------------------------------------------------------------------===//
// Binding access helpers
//===----------------------------------------------------------------------===//

// Resolves binding |ordinal| (plus a |byte_offset| sub-region into it) to a
// pointer, validating it is in-range and the buffer holds the offset window.
// The offset lets one bound buffer carry several tensors (IREE packs constants
// this way) while each command addresses its own sub-region.
static iree_status_t iree_hal_emunpu_hw_resolve_binding(
    int32_t ordinal, uint32_t byte_offset, iree_host_size_t binding_count,
    void* const* bindings, const iree_host_size_t* binding_lengths,
    iree_host_size_t required_bytes, void** out_ptr) {
  *out_ptr = NULL;
  if (ordinal < 0 || (iree_host_size_t)ordinal >= binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command references binding ordinal %d but only %" PRIhsz
        " bindings are bound",
        ordinal, binding_count);
  }
  if (binding_lengths[ordinal] <
      (iree_host_size_t)byte_offset + required_bytes) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "binding %d is %" PRIhsz
        " bytes but the command requires offset %u + %" PRIhsz,
        ordinal, binding_lengths[ordinal], byte_offset, required_bytes);
  }
  *out_ptr = (uint8_t*)bindings[ordinal] + byte_offset;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_emunpu_dtype_size(
    iree_hal_emunpu_dtype_t dtype) {
  switch (dtype) {
    case IREE_HAL_EMUNPU_DTYPE_F32:
      return sizeof(float);
    case IREE_HAL_EMUNPU_DTYPE_I32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}

//===----------------------------------------------------------------------===//
// Compute kernels (the emulated NPU engine)
//===----------------------------------------------------------------------===//

// Maps a simulator status onto an iree_status_t.
static iree_status_t iree_hal_emunpu_from_sim_status(emunpu_status_t status) {
  switch (status) {
    case EMUNPU_OK:
      return iree_ok_status();
    case EMUNPU_ERROR_OUT_OF_MEMORY:
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "emunpu device out of memory");
    case EMUNPU_ERROR_OUT_OF_RANGE:
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "emunpu out of range");
    case EMUNPU_ERROR_UNIMPLEMENTED:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "emunpu unimplemented");
    default:
      return iree_make_status(IREE_STATUS_INTERNAL, "emunpu error %s",
                              emunpu_status_name(status));
  }
}

// The process-wide multi-core simulator device (created lazily on the first
// op). Hoisted to file scope so the at-exit drain handler below can report the
// device's final (freed) occupancy to the visualizer.
static emunpu_device_t* g_emunpu_hw_device = NULL;

// At process exit every op has completed and freed its buffers, so the device's
// live occupancy is back to (near) zero. The MEM events emitted during a run all
// fire *while* an op's buffers are still allocated (see emunpu_run_all_cores), so
// the visualizer's last-known occupancy is the final op's peak and its memory bar
// would otherwise stay pegged forever. Emitting the true post-run occupancy here
// drains the bar once the workload ends.
static void iree_hal_emunpu_hw_viz_drain(void) {
  if (!g_emunpu_hw_device) return;
  emunpu_global_mem_t* gm = emunpu_device_global_mem(g_emunpu_hw_device);
  emunpu_viz_mem(gm->bytes_in_use, gm->capacity);
}

// Returns the process-wide multi-core simulator device, creating it lazily. The
// device hosts all compute (matmul + elementwise) routed off the HAL path. It
// is single-threaded, matching the local-sync execution model that drives
// dispatches here. Both the matmul and elementwise paths share this one device
// (and its 10 cores), so the visualizer shows their combined activity.
//
// Global memory is a *fixed* hardware resource of IREE_EMUNPU_GLOBAL_MEM_BYTES
// (4MB) — not sized to the workload — so the visualizer's occupancy bar reads
// against a stable denominator (e.g. "64.5 KB / 4.0 MB") like a real
// accelerator's on-device memory. The only reason to (re)create the device is a
// single op whose working set (|need|) exceeds the fixed size; in that case the
// arena grows just enough to fit that op and stays grown for later ops.
static iree_status_t iree_hal_emunpu_hw_device(uint64_t need,
                                               emunpu_device_t** out_device) {
  uint64_t capacity = need > IREE_EMUNPU_GLOBAL_MEM_BYTES
                          ? need
                          : IREE_EMUNPU_GLOBAL_MEM_BYTES;
  if (!g_emunpu_hw_device ||
      emunpu_device_global_mem(g_emunpu_hw_device)->capacity < capacity) {
    if (g_emunpu_hw_device) emunpu_device_destroy(g_emunpu_hw_device);
    g_emunpu_hw_device = NULL;
    emunpu_status_t s = emunpu_device_create(capacity, &g_emunpu_hw_device);
    if (s != EMUNPU_OK) return iree_hal_emunpu_from_sim_status(s);
    static int drain_registered = 0;
    if (!drain_registered) {
      atexit(iree_hal_emunpu_hw_viz_drain);
      drain_registered = 1;
    }
  }
  *out_device = g_emunpu_hw_device;
  return iree_ok_status();
}

// Runs a matmul on the multi-core NPU simulator by REPLAYING the compiler's tile
// schedule (|tiles|): each MMA_DIM x MMA_DIM tile's DMA -> MMA16 -> DMA stream is
// issued to the core the compiler assigned it, giving the realistic multi-core
// execution while still computing on the dispatch's binding buffers. The runtime
// performs no tiling decision -- that is now a compile-time concern (see
// emunpu_isa.h). Computing through the device's own global memory (upload ->
// compute -> download) also sidesteps any input/output buffer aliasing IREE may
// introduce, since the inputs are staged before any output is written.
static iree_status_t iree_hal_emunpu_hw_multicore_matmul(
    const void* lhs, const void* rhs, void* out, uint32_t M, uint32_t N,
    uint32_t K, emunpu_dtype_t dtype, int transpose_b,
    const emunpu_gemm_tile_t* tiles, uint32_t tile_count) {
  const size_t elem = emunpu_dtype_size(dtype);
  // Exact live working set: lhs (MxK) + rhs (KxN) + out (MxN). This is what the
  // op occupies in the fixed 4MB global memory (iree_hal_emunpu_hw_device); it
  // only forces a larger arena if it alone exceeds that fixed size.
  const uint64_t need =
      ((uint64_t)M * K + (uint64_t)K * N + (uint64_t)M * N) * elem;
  emunpu_device_t* dev = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_device(need, &dev));

  uint32_t sa[2] = {M, K};
  uint32_t sb[2] = {transpose_b ? N : K, transpose_b ? K : N};
  uint32_t sc[2] = {M, N};
  emunpu_tensor_id_t a = 0, b = 0, c = 0;
  emunpu_status_t s = EMUNPU_OK;
  if ((s = emunpu_device_alloc(dev, dtype, 2, sa, "lhs", &a)) == EMUNPU_OK &&
      (s = emunpu_device_alloc(dev, dtype, 2, sb, "rhs", &b)) == EMUNPU_OK &&
      (s = emunpu_device_alloc(dev, dtype, 2, sc, "out", &c)) == EMUNPU_OK &&
      (s = emunpu_device_upload(dev, a, lhs, (uint64_t)M * K * elem)) ==
          EMUNPU_OK &&
      (s = emunpu_device_upload(dev, b, rhs, (uint64_t)K * N * elem)) ==
          EMUNPU_OK &&
      (s = emunpu_matmul_scheduled(dev, c, a, b, M, N, K, transpose_b, tiles,
                                   tile_count)) == EMUNPU_OK) {
    s = emunpu_device_download(dev, c, out, (uint64_t)M * N * elem);
  }
  // Free in reverse so the arena free-list coalesces back to empty for reuse.
  if (c) emunpu_device_free(dev, c);
  if (b) emunpu_device_free(dev, b);
  if (a) emunpu_device_free(dev, a);
  return iree_hal_emunpu_from_sim_status(s);
}

// Decodes the compiler's tile schedule (the ISA tile records the MATMUL command
// points at) into the simulator's IREE-independent tile form, validating each
// record against the matmul's shape so a malformed blob can't address out of
// bounds. Allocates *|out_tiles| (caller frees); writes NULL/0 for a 0-tile
// (empty) matmul.
static iree_status_t iree_hal_emunpu_hw_decode_tiles(
    const iree_hal_emunpu_matmul_tile_t* tiles, uint32_t first_tile,
    uint32_t tile_count, uint32_t M, uint32_t N, uint32_t K,
    emunpu_gemm_tile_t** out_tiles) {
  *out_tiles = NULL;
  if (tile_count == 0) return iree_ok_status();
  if (!tiles) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "matmul command references %u tiles but the "
                            "executable carries no tile schedule",
                            tile_count);
  }
  emunpu_gemm_tile_t* decoded =
      (emunpu_gemm_tile_t*)malloc((size_t)tile_count * sizeof(*decoded));
  if (!decoded) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to allocate %u matmul tiles", tile_count);
  }
  for (uint32_t i = 0; i < tile_count; ++i) {
    const iree_hal_emunpu_matmul_tile_t* t = &tiles[first_tile + i];
    if (t->tm > IREE_EMUNPU_MMA_DIM || t->tn > IREE_EMUNPU_MMA_DIM ||
        t->tk > IREE_EMUNPU_MMA_DIM || (uint64_t)t->m0 + t->tm > M ||
        (uint64_t)t->n0 + t->tn > N || (uint64_t)t->k0 + t->tk > K) {
      free(decoded);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "matmul tile %u (m0=%u,n0=%u,k0=%u,%ux%ux%u) is out of range for a "
          "%ux%ux%u matmul",
          i, t->m0, t->n0, t->k0, t->tm, t->tn, t->tk, M, N, K);
    }
    decoded[i].core_id = t->core_id;
    decoded[i].m0 = t->m0;
    decoded[i].n0 = t->n0;
    decoded[i].k0 = t->k0;
    decoded[i].tm = t->tm;
    decoded[i].tn = t->tn;
    decoded[i].tk = t->tk;
    decoded[i].accumulate = (t->flags & IREE_HAL_EMUNPU_TILE_ACCUMULATE) != 0;
    decoded[i].writeback = (t->flags & IREE_HAL_EMUNPU_TILE_WRITEBACK) != 0;
  }
  *out_tiles = decoded;
  return iree_ok_status();
}

static iree_status_t iree_hal_emunpu_hw_matmul(
    const iree_hal_emunpu_command_t* command,
    const iree_hal_emunpu_matmul_tile_t* tiles, iree_host_size_t binding_count,
    void* const* bindings, const iree_host_size_t* binding_lengths) {
  const uint32_t M = command->dims[0];
  const uint32_t N = command->dims[1];
  const uint32_t K = command->dims[2];
  const iree_host_size_t elem_size = iree_hal_emunpu_dtype_size(command->dtype);
  if (elem_size == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported matmul element type %u",
                            command->dtype);
  }
  void* lhs = NULL;
  void* rhs = NULL;
  void* out = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
      command->operand_bindings[0], command->operand_offsets[0], binding_count,
      bindings, binding_lengths, (iree_host_size_t)M * K * elem_size, &lhs));
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
      command->operand_bindings[1], command->operand_offsets[1], binding_count,
      bindings, binding_lengths, (iree_host_size_t)K * N * elem_size, &rhs));
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
      command->result_binding, command->result_offset, binding_count, bindings,
      binding_lengths, (iree_host_size_t)M * N * elem_size, &out));

  const bool transpose_b =
      (command->dims[3] & IREE_HAL_EMUNPU_MATMUL_TRANSPOSE_B) != 0;

  // Decode the compile-time tile schedule and hand it to the multi-core
  // simulator, which replays it across the cores (no runtime tiling decision).
  emunpu_gemm_tile_t* sched = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_decode_tiles(
      tiles, command->first_tile, command->tile_count, M, N, K, &sched));

  emunpu_dtype_t dtype = command->dtype == IREE_HAL_EMUNPU_DTYPE_F32
                             ? EMUNPU_DT_FP32
                             : EMUNPU_DT_INT32;
  iree_status_t status = iree_hal_emunpu_hw_multicore_matmul(
      lhs, rhs, out, M, N, K, dtype, transpose_b ? 1 : 0, sched,
      command->tile_count);
  free(sched);
  return status;
}

// Runs an elementwise op (add/relu/copy) on the multi-core simulator. The
// device splits the |count| elements across the cores (unary relu/copy and
// equal-shape add fan out across all of them; a broadcast add -- period != 0,
// the nn.Linear row bias -- runs on a single core so the global broadcast index
// stays consistent). |rhs| is NULL for the unary ops.
static iree_status_t iree_hal_emunpu_hw_multicore_elementwise(
    iree_hal_emunpu_opcode_t opcode, emunpu_dtype_t dtype, uint32_t count,
    uint32_t period, const void* lhs, const void* rhs, void* out) {
  const size_t elem = emunpu_dtype_size(dtype);
  const int binary = (rhs != NULL);
  const uint32_t rhs_count = period ? period : count;
  // Exact live working set: a (count) + optional b (rhs_count) + c (count).
  // Occupies the fixed 4MB global memory (iree_hal_emunpu_hw_device).
  const uint64_t need =
      ((uint64_t)count + (binary ? rhs_count : 0) + count) * elem;
  emunpu_device_t* dev = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_device(need, &dev));

  emunpu_ew_kind_t kind;
  switch (opcode) {
    case IREE_HAL_EMUNPU_OP_ADD:
      kind = EMUNPU_EW_ADD;
      break;
    case IREE_HAL_EMUNPU_OP_RELU:
      kind = EMUNPU_EW_RELU;
      break;
    case IREE_HAL_EMUNPU_OP_COPY:
      kind = EMUNPU_EW_COPY;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported elementwise opcode %u", opcode);
  }

  uint32_t na = count, nb = rhs_count, nc = count;
  emunpu_tensor_id_t a = 0, b = 0, c = 0;
  emunpu_status_t s = EMUNPU_OK;
  if ((s = emunpu_device_alloc(dev, dtype, 1, &na, "ew_a", &a)) == EMUNPU_OK &&
      (!binary || (s = emunpu_device_alloc(dev, dtype, 1, &nb, "ew_b", &b)) ==
                      EMUNPU_OK) &&
      (s = emunpu_device_alloc(dev, dtype, 1, &nc, "ew_c", &c)) == EMUNPU_OK &&
      (s = emunpu_device_upload(dev, a, lhs, (uint64_t)count * elem)) ==
          EMUNPU_OK &&
      (!binary ||
       (s = emunpu_device_upload(dev, b, rhs, (uint64_t)rhs_count * elem)) ==
           EMUNPU_OK) &&
      (s = emunpu_vector_op(dev, NULL, c, a, b, count, kind, period)) ==
          EMUNPU_OK) {
    s = emunpu_device_download(dev, c, out, (uint64_t)count * elem);
  }
  if (c) emunpu_device_free(dev, c);
  if (b) emunpu_device_free(dev, b);
  if (a) emunpu_device_free(dev, a);
  return iree_hal_emunpu_from_sim_status(s);
}

// Shared driver for the elementwise ops (add/relu/copy).
static iree_status_t iree_hal_emunpu_hw_elementwise(
    const iree_hal_emunpu_command_t* command, iree_host_size_t binding_count,
    void* const* bindings, const iree_host_size_t* binding_lengths) {
  const uint32_t count = command->dims[0];
  const iree_host_size_t elem_size = iree_hal_emunpu_dtype_size(command->dtype);
  if (elem_size == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported elementwise element type %u",
                            command->dtype);
  }
  const iree_host_size_t bytes = (iree_host_size_t)count * elem_size;
  const bool binary = command->opcode == IREE_HAL_EMUNPU_OP_ADD;
  // For ADD, dims[1] is the rhs broadcast period (0 => equal-shape). A nonzero
  // period means rhs holds only `period` elements indexed as rhs[i % period]
  // (the nn.Linear row-bias pattern).
  const uint32_t rhs_period =
      command->opcode == IREE_HAL_EMUNPU_OP_ADD ? command->dims[1] : 0;
  const iree_host_size_t rhs_bytes =
      (rhs_period ? (iree_host_size_t)rhs_period : (iree_host_size_t)count) *
      elem_size;

  void* lhs = NULL;
  void* rhs = NULL;
  void* out = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
      command->operand_bindings[0], command->operand_offsets[0], binding_count,
      bindings, binding_lengths, bytes, &lhs));
  if (binary) {
    IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
        command->operand_bindings[1], command->operand_offsets[1],
        binding_count, bindings, binding_lengths, rhs_bytes, &rhs));
  }
  IREE_RETURN_IF_ERROR(iree_hal_emunpu_hw_resolve_binding(
      command->result_binding, command->result_offset, binding_count, bindings,
      binding_lengths, bytes, &out));

  // Hand the elementwise op to the multi-core simulator, which splits the work
  // across the cores (see iree_hal_emunpu_hw_multicore_elementwise).
  emunpu_dtype_t dtype = command->dtype == IREE_HAL_EMUNPU_DTYPE_F32
                             ? EMUNPU_DT_FP32
                             : EMUNPU_DT_INT32;
  return iree_hal_emunpu_hw_multicore_elementwise(
      command->opcode, dtype, count, rhs_period, lhs, binary ? rhs : NULL, out);
}

iree_status_t iree_hal_emunpu_hw_execute_command(
    const iree_hal_emunpu_command_t* command,
    const iree_hal_emunpu_matmul_tile_t* tiles, iree_host_size_t binding_count,
    void* const* bindings, const iree_host_size_t* binding_lengths) {
  switch (command->opcode) {
    case IREE_HAL_EMUNPU_OP_MATMUL:
      return iree_hal_emunpu_hw_matmul(command, tiles, binding_count, bindings,
                                       binding_lengths);
    case IREE_HAL_EMUNPU_OP_ADD:
    case IREE_HAL_EMUNPU_OP_RELU:
    case IREE_HAL_EMUNPU_OP_COPY:
      return iree_hal_emunpu_hw_elementwise(command, binding_count, bindings,
                                            binding_lengths);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unknown emunpu opcode %u", command->opcode);
  }
}
