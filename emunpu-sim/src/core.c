// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emunpu/core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emunpu/viz.h"

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_core_initialize(emunpu_core_t* core, uint32_t core_id) {
  if (!core) return EMUNPU_ERROR_INVALID_ARGUMENT;
  memset(core, 0, sizeof(*core));
  core->core_id = core_id;
  core->local_capacity = IREE_EMUNPU_LOCAL_MEM_BYTES;
  core->local_mem = (uint8_t*)malloc((size_t)core->local_capacity);
  if (!core->local_mem) return EMUNPU_ERROR_OUT_OF_MEMORY;
  core->queue_cap = 64;
  core->queue =
      (emunpu_instr_t*)malloc(core->queue_cap * sizeof(emunpu_instr_t));
  if (!core->queue) {
    free(core->local_mem);
    core->local_mem = NULL;
    return EMUNPU_ERROR_OUT_OF_MEMORY;
  }
  return EMUNPU_OK;
}

void emunpu_core_deinitialize(emunpu_core_t* core) {
  if (!core) return;
  free(core->local_mem);
  free(core->queue);
  memset(core, 0, sizeof(*core));
}

emunpu_status_t emunpu_core_local_alloc(emunpu_core_t* core, uint64_t bytes,
                                        uint64_t* out_offset) {
  // Align local tiles to 64B for clean tile loads.
  uint64_t base = (core->local_watermark + 63) & ~(uint64_t)63;
  if (base + bytes > core->local_capacity) return EMUNPU_ERROR_OUT_OF_MEMORY;
  *out_offset = base;
  core->local_watermark = base + bytes;
  return EMUNPU_OK;
}

void emunpu_core_local_reset(emunpu_core_t* core) { core->local_watermark = 0; }

emunpu_status_t emunpu_core_enqueue(emunpu_core_t* core,
                                    const emunpu_instr_t* instr) {
  if (!core || !instr) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (core->queue_len == core->queue_cap) {
    uint32_t new_cap = core->queue_cap * 2;
    emunpu_instr_t* grown =
        (emunpu_instr_t*)realloc(core->queue, new_cap * sizeof(emunpu_instr_t));
    if (!grown) return EMUNPU_ERROR_OUT_OF_MEMORY;
    core->queue = grown;
    core->queue_cap = new_cap;
  }
  core->queue[core->queue_len++] = *instr;
  return EMUNPU_OK;
}

void emunpu_core_queue_reset(emunpu_core_t* core) { core->queue_len = 0; }

//===----------------------------------------------------------------------===//
// Operand resolution
//===----------------------------------------------------------------------===//

// Resolves a (space, byte offset) operand to a host pointer, validating that
// the [offset, offset + required_bytes) window is in-range for its space.
static emunpu_status_t emunpu_core_resolve(
    emunpu_core_t* core, emunpu_space_t space, uint64_t offset,
    uint8_t* global_arena, uint64_t global_capacity, uint64_t required_bytes,
    uint8_t** out_ptr) {
  if (space == EMUNPU_SPACE_LOCAL) {
    if (offset + required_bytes > core->local_capacity) {
      return EMUNPU_ERROR_OUT_OF_RANGE;
    }
    *out_ptr = core->local_mem + offset;
    return EMUNPU_OK;
  }
  if (offset + required_bytes > global_capacity) {
    return EMUNPU_ERROR_OUT_OF_RANGE;
  }
  *out_ptr = global_arena + offset;
  return EMUNPU_OK;
}

//===----------------------------------------------------------------------===//
// Trace
//===----------------------------------------------------------------------===//

static void emunpu_core_trace(const emunpu_core_t* core,
                              const emunpu_instr_t* in) {
  switch (in->op) {
    case EMUNPU_I_MMA16:
      fprintf(stderr, "[core%u] MMA16 %s*%s->%s M=%u N=%u K=%u %s%s\n",
              core->core_id, emunpu_dtype_name(in->a_dtype),
              emunpu_dtype_name(in->b_dtype), emunpu_dtype_name(in->c_dtype),
              in->M, in->N, in->K,
              (in->flags & EMUNPU_MMA_TRANSPOSE_B) ? "transpose_b " : "",
              (in->flags & EMUNPU_MMA_ACCUMULATE) ? "acc+=" : "acc=");
      break;
    case EMUNPU_I_DMA_G2L:
    case EMUNPU_I_DMA_L2G:
      fprintf(stderr, "[core%u] %s %s rows=%u cols=%u (%llu->%llu B)\n",
              core->core_id, emunpu_iop_name(in->op),
              emunpu_dtype_name(in->c_dtype), in->M, in->N,
              (unsigned long long)in->a_offset,
              (unsigned long long)in->c_offset);
      break;
    case EMUNPU_I_EW:
      fprintf(stderr, "[core%u] EW.%s %s n=%u period=%u\n", core->core_id,
              emunpu_ew_name((emunpu_ew_kind_t)in->flags),
              emunpu_dtype_name(in->c_dtype), in->M, in->bcast_period);
      break;
    case EMUNPU_I_REDUCE:
      fprintf(stderr, "[core%u] REDUCE.%s %s rows=%u width=%u\n", core->core_id,
              emunpu_reduce_name((emunpu_reduce_kind_t)in->flags),
              emunpu_dtype_name(in->c_dtype), in->M, in->N);
      break;
    case EMUNPU_I_ACT:
      fprintf(stderr, "[core%u] ACT.%s %s n=%u alpha=%g beta=%g\n",
              core->core_id, emunpu_act_name((emunpu_act_kind_t)in->flags),
              emunpu_dtype_name(in->c_dtype), in->M, in->param0, in->param1);
      break;
    case EMUNPU_I_SOFTMAX:
      fprintf(stderr, "[core%u] SOFTMAX%s %s rows=%u width=%u\n", core->core_id,
              (in->flags & EMUNPU_SOFTMAX_LOG) ? ".log" : "",
              emunpu_dtype_name(in->c_dtype), in->M, in->N);
      break;
    case EMUNPU_I_NORM:
      fprintf(stderr, "[core%u] NORM.%s %s rows=%u width=%u eps=%g\n",
              core->core_id, emunpu_norm_name((emunpu_norm_kind_t)in->flags),
              emunpu_dtype_name(in->c_dtype), in->M, in->N, in->param0);
      break;
    case EMUNPU_I_TRANSPOSE:
      fprintf(stderr, "[core%u] TRANSPOSE %s %ux%u->%ux%u\n", core->core_id,
              emunpu_dtype_name(in->c_dtype), in->M, in->N, in->N, in->M);
      break;
    case EMUNPU_I_IM2COL:
      fprintf(
          stderr,
          "[core%u] IM2COL %s rows=%u@%u width=%u k=%ux%u s=%ux%u p=%ux%u\n",
          core->core_id, emunpu_dtype_name(in->c_dtype), in->M,
          in->spatial.first_out_row, in->N, in->spatial.kernel_h,
          in->spatial.kernel_w, in->spatial.stride_h, in->spatial.stride_w,
          in->spatial.pad_h, in->spatial.pad_w);
      break;
    case EMUNPU_I_POOL2D:
      fprintf(stderr, "[core%u] POOL2D.%s %s rows=%u@%u %ux%u->%ux%u k=%ux%u\n",
              core->core_id, emunpu_pool_name((emunpu_pool_kind_t)in->flags),
              emunpu_dtype_name(in->c_dtype), in->M, in->spatial.first_out_row,
              in->spatial.in_h, in->spatial.in_w, in->spatial.out_h,
              in->spatial.out_w, in->spatial.kernel_h, in->spatial.kernel_w);
      break;
    case EMUNPU_I_GATHER:
      fprintf(stderr, "[core%u] GATHER %s rows=%u width=%u table_rows=%u\n",
              core->core_id, emunpu_dtype_name(in->c_dtype), in->M, in->N,
              in->K);
      break;
    default:
      fprintf(stderr, "[core%u] %s %s n=%u\n", core->core_id,
              emunpu_iop_name(in->op), emunpu_dtype_name(in->c_dtype), in->M);
      break;
  }
}

//===----------------------------------------------------------------------===//
// Compute kernels
//===----------------------------------------------------------------------===//

// 2D (possibly strided) byte copy between memory spaces. M rows of N elements;
// a_* is the source, c_* the destination. Row strides default to a packed row.
static emunpu_status_t emunpu_core_dma(emunpu_core_t* core,
                                       const emunpu_instr_t* in, uint8_t* g,
                                       uint64_t gcap) {
  size_t elem = emunpu_dtype_size(in->c_dtype);
  if (elem == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint32_t rows = in->M ? in->M : 1;
  uint32_t cols = in->N;
  uint64_t row_bytes = (uint64_t)cols * elem;
  uint64_t src_stride = in->a_row_stride ? in->a_row_stride : row_bytes;
  uint64_t dst_stride = in->c_row_stride ? in->c_row_stride : row_bytes;

  emunpu_space_t src_space =
      (in->op == EMUNPU_I_DMA_G2L) ? EMUNPU_SPACE_GLOBAL : EMUNPU_SPACE_LOCAL;
  emunpu_space_t dst_space =
      (in->op == EMUNPU_I_DMA_G2L) ? EMUNPU_SPACE_LOCAL : EMUNPU_SPACE_GLOBAL;
  uint8_t* src = NULL;
  uint8_t* dst = NULL;
  emunpu_status_t st =
      emunpu_core_resolve(core, src_space, in->a_offset, g, gcap,
                          (uint64_t)(rows - 1) * src_stride + row_bytes, &src);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, dst_space, in->c_offset, g, gcap,
                           (uint64_t)(rows - 1) * dst_stride + row_bytes, &dst);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    memcpy(dst + r * dst_stride, src + r * src_stride, row_bytes);
  }
  core->dma_bytes += (uint64_t)rows * row_bytes;
  return EMUNPU_OK;
}

// 16x16-class tile multiply-accumulate. C[M x N] (+)= A[M x K] * B[K x N], with
// inputs read in their own dtypes (mixed precision) and accumulated in the
// accumulator's wider precision (double here; bit-faithful for the modeled
// fp32/i32 accumulate). transpose_b reads B as N x K.
static emunpu_status_t emunpu_core_mma16(emunpu_core_t* core,
                                         const emunpu_instr_t* in, uint8_t* g,
                                         uint64_t gcap) {
  // Tile dimensions are bounded by the MMA engine's width; emunpu_core_execute
  // has already enforced that (emunpu_instr_validate).
  uint32_t M = in->M, N = in->N, K = in->K;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t bs = emunpu_dtype_size(in->b_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !bs || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;
  int transpose_b = (in->flags & EMUNPU_MMA_TRANSPOSE_B) != 0;
  int accumulate = (in->flags & EMUNPU_MMA_ACCUMULATE) != 0;

  uint64_t a_stride = in->a_row_stride ? in->a_row_stride : (uint64_t)K * as;
  uint64_t b_stride = in->b_row_stride ? in->b_row_stride
                                       : (uint64_t)(transpose_b ? K : N) * bs;
  uint64_t c_stride = in->c_row_stride ? in->c_row_stride : (uint64_t)N * cs;

  uint8_t *A = NULL, *B = NULL, *C = NULL;
  emunpu_status_t st =
      emunpu_core_resolve(core, in->a_space, in->a_offset, g, gcap,
                          (uint64_t)(M - 1) * a_stride + (uint64_t)K * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->b_space, in->b_offset, g, gcap,
                           (uint64_t)((transpose_b ? N : K) - 1) * b_stride +
                               (uint64_t)(transpose_b ? K : N) * bs,
                           &B);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)(M - 1) * c_stride + (uint64_t)N * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < M; ++i) {
    const uint8_t* a_row = A + i * a_stride;
    uint8_t* c_row = C + i * c_stride;
    for (uint32_t j = 0; j < N; ++j) {
      double acc = accumulate ? emunpu_dtype_load(c_row, j, in->c_dtype) : 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        double av = emunpu_dtype_load(a_row, k, in->a_dtype);
        // B is K x N (b[k,j]) or, transposed, N x K (b[j,k]).
        const uint8_t* b_row =
            transpose_b ? (B + j * b_stride) : (B + k * b_stride);
        double bv = emunpu_dtype_load(b_row, transpose_b ? k : j, in->b_dtype);
        acc += av * bv;
      }
      emunpu_dtype_store(c_row, j, in->c_dtype, acc);
    }
  }
  core->mma_count++;
  return EMUNPU_OK;
}

// y[M] (+)= A[M x K] * x[K].
static emunpu_status_t emunpu_core_matvec(emunpu_core_t* core,
                                          const emunpu_instr_t* in, uint8_t* g,
                                          uint64_t gcap) {
  uint32_t M = in->M, K = in->K;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t bs = emunpu_dtype_size(in->b_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !bs || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;
  int accumulate = (in->flags & EMUNPU_MMA_ACCUMULATE) != 0;
  uint64_t a_stride = in->a_row_stride ? in->a_row_stride : (uint64_t)K * as;

  uint8_t *A = NULL, *X = NULL, *Y = NULL;
  emunpu_status_t st =
      emunpu_core_resolve(core, in->a_space, in->a_offset, g, gcap,
                          (uint64_t)(M - 1) * a_stride + (uint64_t)K * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->b_space, in->b_offset, g, gcap,
                           (uint64_t)K * bs, &X);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)M * cs, &Y);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < M; ++i) {
    const uint8_t* a_row = A + i * a_stride;
    double acc = accumulate ? emunpu_dtype_load(Y, i, in->c_dtype) : 0.0;
    for (uint32_t k = 0; k < K; ++k) {
      acc += emunpu_dtype_load(a_row, k, in->a_dtype) *
             emunpu_dtype_load(X, k, in->b_dtype);
    }
    emunpu_dtype_store(Y, i, in->c_dtype, acc);
  }
  return EMUNPU_OK;
}

// Elementwise unary/binary op over M elements. Binary ops broadcast operand b
// with period bcast_period (0 => equal-shape).
static emunpu_status_t emunpu_core_ew(emunpu_core_t* core,
                                      const emunpu_instr_t* in, uint8_t* g,
                                      uint64_t gcap) {
  uint32_t n = in->M;
  emunpu_ew_kind_t kind = (emunpu_ew_kind_t)in->flags;
  int binary = emunpu_ew_is_binary(kind);
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;

  uint8_t *A = NULL, *B = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(core, in->a_space, in->a_offset, g,
                                           gcap, (uint64_t)n * as, &A);
  if (st != EMUNPU_OK) return st;
  uint32_t period = in->bcast_period;
  if (binary) {
    size_t bs = emunpu_dtype_size(in->b_dtype);
    uint64_t b_elems = period ? period : n;
    st = emunpu_core_resolve(core, in->b_space, in->b_offset, g, gcap,
                             b_elems * bs, &B);
    if (st != EMUNPU_OK) return st;
  }
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)n * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < n; ++i) {
    double a = emunpu_dtype_load(A, i, in->a_dtype);
    double r;
    if (binary) {
      // Broadcast indexing uses the *global* element index, which for a slice
      // of a larger op is bcast_phase + i.
      uint32_t bi = period ? (in->bcast_phase + i) % period : i;
      double b = emunpu_dtype_load(B, bi, in->b_dtype);
      switch (kind) {
        case EMUNPU_EW_ADD:
          r = a + b;
          break;
        case EMUNPU_EW_SUB:
          r = a - b;
          break;
        case EMUNPU_EW_MUL:
          r = a * b;
          break;
        case EMUNPU_EW_DIV:
          r = a / b;
          break;
        case EMUNPU_EW_MAX:
          r = a > b ? a : b;
          break;
        case EMUNPU_EW_MIN:
          r = a < b ? a : b;
          break;
        case EMUNPU_EW_POW:
          r = pow(a, b);
          break;
        default:
          r = a;
          break;
      }
    } else {
      r = (kind == EMUNPU_EW_RELU) ? (a > 0.0 ? a : 0.0) : a;  // relu / copy
    }
    emunpu_dtype_store(C, i, in->c_dtype, r);
  }
  return EMUNPU_OK;
}

// Reduction of each of M rows of width N to a single output element.
static emunpu_status_t emunpu_core_reduce(emunpu_core_t* core,
                                          const emunpu_instr_t* in, uint8_t* g,
                                          uint64_t gcap) {
  uint32_t rows = in->M ? in->M : 1;
  uint32_t width = in->N;
  emunpu_reduce_kind_t kind = (emunpu_reduce_kind_t)in->flags;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || width == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint64_t a_stride =
      in->a_row_stride ? in->a_row_stride : (uint64_t)width * as;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(
      core, in->a_space, in->a_offset, g, gcap,
      (uint64_t)(rows - 1) * a_stride + (uint64_t)width * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)rows * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    const uint8_t* row = A + r * a_stride;
    double acc = emunpu_dtype_load(row, 0, in->a_dtype);
    uint32_t arg = 0;  // index of the running extremum (ARGMAX / ARGMIN)
    for (uint32_t j = 1; j < width; ++j) {
      double v = emunpu_dtype_load(row, j, in->a_dtype);
      switch (kind) {
        case EMUNPU_REDUCE_SUM:
        case EMUNPU_REDUCE_MEAN:
          acc += v;
          break;
        case EMUNPU_REDUCE_PROD:
          acc *= v;
          break;
        case EMUNPU_REDUCE_MAX:
          acc = v > acc ? v : acc;
          break;
        case EMUNPU_REDUCE_MIN:
          acc = v < acc ? v : acc;
          break;
        case EMUNPU_REDUCE_ARGMAX:
          if (v > acc) {
            acc = v;
            arg = j;  // first occurrence wins (strict >), as in PyTorch
          }
          break;
        case EMUNPU_REDUCE_ARGMIN:
          if (v < acc) {
            acc = v;
            arg = j;
          }
          break;
        default:
          return EMUNPU_ERROR_UNIMPLEMENTED;
      }
    }
    if (kind == EMUNPU_REDUCE_MEAN) acc /= (double)width;
    // ARG* reductions emit the *position* of the extremum, not its value.
    if (kind == EMUNPU_REDUCE_ARGMAX || kind == EMUNPU_REDUCE_ARGMIN) {
      acc = (double)arg;
    }
    emunpu_dtype_store(C, r, in->c_dtype, acc);
  }
  return EMUNPU_OK;
}

// Fused multiply-add: c[i] = a[i] * b[i] + c[i].
static emunpu_status_t emunpu_core_fma(emunpu_core_t* core,
                                       const emunpu_instr_t* in, uint8_t* g,
                                       uint64_t gcap) {
  uint32_t n = in->M;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t bs = emunpu_dtype_size(in->b_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !bs || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint8_t *A = NULL, *B = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(core, in->a_space, in->a_offset, g,
                                           gcap, (uint64_t)n * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->b_space, in->b_offset, g, gcap,
                           (uint64_t)n * bs, &B);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)n * cs, &C);
  if (st != EMUNPU_OK) return st;
  for (uint32_t i = 0; i < n; ++i) {
    double v = emunpu_dtype_load(A, i, in->a_dtype) *
                   emunpu_dtype_load(B, i, in->b_dtype) +
               emunpu_dtype_load(C, i, in->c_dtype);
    emunpu_dtype_store(C, i, in->c_dtype, v);
  }
  return EMUNPU_OK;
}

// Datatype conversion: c[i] = convert(a[i]) from a_dtype to c_dtype.
static emunpu_status_t emunpu_core_convert(emunpu_core_t* core,
                                           const emunpu_instr_t* in, uint8_t* g,
                                           uint64_t gcap) {
  uint32_t n = in->M;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(core, in->a_space, in->a_offset, g,
                                           gcap, (uint64_t)n * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)n * cs, &C);
  if (st != EMUNPU_OK) return st;
  for (uint32_t i = 0; i < n; ++i) {
    emunpu_dtype_store(C, i, in->c_dtype, emunpu_dtype_load(A, i, in->a_dtype));
  }
  return EMUNPU_OK;
}

// Activation unit: c[i] = act(a[i]) over M elements, with the kind's scalar
// parameters in param0/param1. Reads and writes through the dtype accessors, so
// an fp16 tensor is widened, transformed in double, and re-narrowed on store.
static emunpu_status_t emunpu_core_act(emunpu_core_t* core,
                                       const emunpu_instr_t* in, uint8_t* g,
                                       uint64_t gcap) {
  uint32_t n = in->M;
  emunpu_act_kind_t kind = (emunpu_act_kind_t)in->flags;
  if (kind >= EMUNPU_ACT_COUNT) return EMUNPU_ERROR_INVALID_ARGUMENT;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs) return EMUNPU_ERROR_INVALID_ARGUMENT;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(core, in->a_space, in->a_offset, g,
                                           gcap, (uint64_t)n * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)n * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < n; ++i) {
    double v = emunpu_act_apply(kind, emunpu_dtype_load(A, i, in->a_dtype),
                                in->param0, in->param1);
    emunpu_dtype_store(C, i, in->c_dtype, v);
  }
  return EMUNPU_OK;
}

// Row-wise softmax over M rows of width N, in the numerically stable
// subtract-the-row-max form (so exp() never overflows regardless of scale).
// EMUNPU_SOFTMAX_LOG emits log(softmax(x)) = x - max - log(sum) instead.
static emunpu_status_t emunpu_core_softmax(emunpu_core_t* core,
                                           const emunpu_instr_t* in, uint8_t* g,
                                           uint64_t gcap) {
  uint32_t rows = in->M ? in->M : 1;
  uint32_t width = in->N;
  int log_variant = (in->flags & EMUNPU_SOFTMAX_LOG) != 0;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || width == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint64_t a_stride =
      in->a_row_stride ? in->a_row_stride : (uint64_t)width * as;
  uint64_t c_stride =
      in->c_row_stride ? in->c_row_stride : (uint64_t)width * cs;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(
      core, in->a_space, in->a_offset, g, gcap,
      (uint64_t)(rows - 1) * a_stride + (uint64_t)width * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(
      core, in->c_space, in->c_offset, g, gcap,
      (uint64_t)(rows - 1) * c_stride + (uint64_t)width * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    const uint8_t* row = A + r * a_stride;
    uint8_t* out = C + r * c_stride;
    double max = emunpu_dtype_load(row, 0, in->a_dtype);
    for (uint32_t j = 1; j < width; ++j) {
      double v = emunpu_dtype_load(row, j, in->a_dtype);
      if (v > max) max = v;
    }
    double sum = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
      sum += exp(emunpu_dtype_load(row, j, in->a_dtype) - max);
    }
    double log_sum = log(sum);
    for (uint32_t j = 0; j < width; ++j) {
      double shifted = emunpu_dtype_load(row, j, in->a_dtype) - max;
      emunpu_dtype_store(out, j, in->c_dtype,
                         log_variant ? shifted - log_sum : exp(shifted) / sum);
    }
  }
  return EMUNPU_OK;
}

// Row-wise normalization over M rows of width N. LAYER removes the row mean and
// divides by the row standard deviation (biased variance, matching
// torch.nn.LayerNorm); RMS divides by the root-mean-square without centering.
// The affine (gamma/beta) term is *not* part of this instruction -- the device
// layer applies it as broadcast EW MUL/ADD, so the unit stays single-operand.
static emunpu_status_t emunpu_core_norm(emunpu_core_t* core,
                                        const emunpu_instr_t* in, uint8_t* g,
                                        uint64_t gcap) {
  uint32_t rows = in->M ? in->M : 1;
  uint32_t width = in->N;
  emunpu_norm_kind_t kind = (emunpu_norm_kind_t)in->flags;
  if (kind >= EMUNPU_NORM_COUNT) return EMUNPU_ERROR_INVALID_ARGUMENT;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || width == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint64_t a_stride =
      in->a_row_stride ? in->a_row_stride : (uint64_t)width * as;
  uint64_t c_stride =
      in->c_row_stride ? in->c_row_stride : (uint64_t)width * cs;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(
      core, in->a_space, in->a_offset, g, gcap,
      (uint64_t)(rows - 1) * a_stride + (uint64_t)width * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(
      core, in->c_space, in->c_offset, g, gcap,
      (uint64_t)(rows - 1) * c_stride + (uint64_t)width * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    const uint8_t* row = A + r * a_stride;
    uint8_t* out = C + r * c_stride;
    double mean = 0.0;
    if (kind == EMUNPU_NORM_LAYER) {
      for (uint32_t j = 0; j < width; ++j) {
        mean += emunpu_dtype_load(row, j, in->a_dtype);
      }
      mean /= (double)width;
    }
    double sq = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
      double d = emunpu_dtype_load(row, j, in->a_dtype) - mean;
      sq += d * d;
    }
    double scale = 1.0 / sqrt(sq / (double)width + in->param0);
    for (uint32_t j = 0; j < width; ++j) {
      double v = (emunpu_dtype_load(row, j, in->a_dtype) - mean) * scale;
      emunpu_dtype_store(out, j, in->c_dtype, v);
    }
  }
  return EMUNPU_OK;
}

// 2D transpose: reads A[M x N] and writes C[N x M]. Both operands may be
// sub-blocks of larger matrices via their row strides, which is how the device
// splits a transpose across cores: core k reads a row band of A and writes the
// matching *column* band of C.
static emunpu_status_t emunpu_core_transpose(emunpu_core_t* core,
                                             const emunpu_instr_t* in,
                                             uint8_t* g, uint64_t gcap) {
  uint32_t M = in->M, N = in->N;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || M == 0 || N == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  uint64_t a_stride = in->a_row_stride ? in->a_row_stride : (uint64_t)N * as;
  uint64_t c_stride = in->c_row_stride ? in->c_row_stride : (uint64_t)M * cs;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st =
      emunpu_core_resolve(core, in->a_space, in->a_offset, g, gcap,
                          (uint64_t)(M - 1) * a_stride + (uint64_t)N * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)(N - 1) * c_stride + (uint64_t)M * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < M; ++i) {
    const uint8_t* a_row = A + i * a_stride;
    for (uint32_t j = 0; j < N; ++j) {
      emunpu_dtype_store(C + j * c_stride, i, in->c_dtype,
                         emunpu_dtype_load(a_row, j, in->a_dtype));
    }
  }
  return EMUNPU_OK;
}

// im2col: turns a CHW image into the patch matrix a convolution needs, so the
// convolution itself becomes a GEMM on the existing MMA engine (exactly how
// real NPUs lower conv2d). Emits M patch rows starting at logical patch row
// spatial.first_out_row; row (oy,ox) holds every input element the output pixel
// at (oy,ox) touches, ordered (channel, kernel_y, kernel_x). Out-of-bounds
// (padding) elements read as zero.
static emunpu_status_t emunpu_core_im2col(emunpu_core_t* core,
                                          const emunpu_instr_t* in, uint8_t* g,
                                          uint64_t gcap) {
  const emunpu_spatial_t* sp = &in->spatial;
  uint32_t rows = in->M;
  uint32_t width = in->N;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || rows == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (!sp->channels || !sp->kernel_h || !sp->kernel_w || !sp->out_h ||
      !sp->out_w) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  if (width != sp->channels * sp->kernel_h * sp->kernel_w) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  if ((uint64_t)sp->first_out_row + rows > (uint64_t)sp->out_h * sp->out_w) {
    return EMUNPU_ERROR_OUT_OF_RANGE;
  }
  uint64_t c_stride =
      in->c_row_stride ? in->c_row_stride : (uint64_t)width * cs;

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(
      core, in->a_space, in->a_offset, g, gcap,
      (uint64_t)sp->channels * sp->in_h * sp->in_w * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(
      core, in->c_space, in->c_offset, g, gcap,
      (uint64_t)(rows - 1) * c_stride + (uint64_t)width * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    uint32_t patch = sp->first_out_row + r;
    uint32_t oy = patch / sp->out_w;
    uint32_t ox = patch % sp->out_w;
    uint8_t* out = C + r * c_stride;
    uint32_t col = 0;
    for (uint32_t ch = 0; ch < sp->channels; ++ch) {
      size_t plane = (size_t)ch * sp->in_h * sp->in_w;
      for (uint32_t ky = 0; ky < sp->kernel_h; ++ky) {
        // Signed intermediates: the window can start left of / above the image.
        int64_t iy = (int64_t)oy * sp->stride_h - sp->pad_h + ky;
        for (uint32_t kx = 0; kx < sp->kernel_w; ++kx, ++col) {
          int64_t ix = (int64_t)ox * sp->stride_w - sp->pad_w + kx;
          double v = 0.0;
          if (iy >= 0 && iy < (int64_t)sp->in_h && ix >= 0 &&
              ix < (int64_t)sp->in_w) {
            v = emunpu_dtype_load(A, plane + (size_t)iy * sp->in_w + (size_t)ix,
                                  in->a_dtype);
          }
          emunpu_dtype_store(out, col, in->c_dtype, v);
        }
      }
    }
  }
  return EMUNPU_OK;
}

// 2D pooling over a row band of ONE CHW plane: emits M output rows starting at
// spatial.first_out_row, each out_w wide. a_offset addresses the input plane;
// c_offset addresses the first output row this instruction writes.
//
// Padding contributes nothing: MAX ignores out-of-bounds taps and AVG divides
// by the in-bounds tap count (PyTorch's count_include_pad=False).
static emunpu_status_t emunpu_core_pool2d(emunpu_core_t* core,
                                          const emunpu_instr_t* in, uint8_t* g,
                                          uint64_t gcap) {
  const emunpu_spatial_t* sp = &in->spatial;
  uint32_t rows = in->M;
  emunpu_pool_kind_t kind = (emunpu_pool_kind_t)in->flags;
  if (kind >= EMUNPU_POOL_COUNT) return EMUNPU_ERROR_INVALID_ARGUMENT;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !cs || rows == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (!sp->kernel_h || !sp->kernel_w || !sp->out_h || !sp->out_w) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  if ((uint64_t)sp->first_out_row + rows > sp->out_h) {
    return EMUNPU_ERROR_OUT_OF_RANGE;
  }

  uint8_t *A = NULL, *C = NULL;
  emunpu_status_t st =
      emunpu_core_resolve(core, in->a_space, in->a_offset, g, gcap,
                          (uint64_t)sp->in_h * sp->in_w * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->c_space, in->c_offset, g, gcap,
                           (uint64_t)rows * sp->out_w * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t r = 0; r < rows; ++r) {
    uint32_t oy = sp->first_out_row + r;
    for (uint32_t ox = 0; ox < sp->out_w; ++ox) {
      double acc = 0.0;
      uint32_t taps = 0;
      for (uint32_t ky = 0; ky < sp->kernel_h; ++ky) {
        int64_t iy = (int64_t)oy * sp->stride_h - sp->pad_h + ky;
        if (iy < 0 || iy >= (int64_t)sp->in_h) continue;
        for (uint32_t kx = 0; kx < sp->kernel_w; ++kx) {
          int64_t ix = (int64_t)ox * sp->stride_w - sp->pad_w + kx;
          if (ix < 0 || ix >= (int64_t)sp->in_w) continue;
          double v = emunpu_dtype_load(A, (size_t)iy * sp->in_w + (size_t)ix,
                                       in->a_dtype);
          if (kind == EMUNPU_POOL_MAX) {
            if (taps == 0 || v > acc) acc = v;
          } else {
            acc += v;
          }
          ++taps;
        }
      }
      double result = 0.0;
      if (taps) result = (kind == EMUNPU_POOL_AVG) ? acc / (double)taps : acc;
      emunpu_dtype_store(C, (size_t)r * sp->out_w + ox, in->c_dtype, result);
    }
  }
  return EMUNPU_OK;
}

// Row gather (embedding lookup): c[i, :] = a[idx[i], :] for M indices held in
// operand b, over a table of K rows and N columns.
static emunpu_status_t emunpu_core_gather(emunpu_core_t* core,
                                          const emunpu_instr_t* in, uint8_t* g,
                                          uint64_t gcap) {
  uint32_t count = in->M, width = in->N, table_rows = in->K;
  size_t as = emunpu_dtype_size(in->a_dtype);
  size_t bs = emunpu_dtype_size(in->b_dtype);
  size_t cs = emunpu_dtype_size(in->c_dtype);
  if (!as || !bs || !cs || !width || !table_rows) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  if (count == 0) return EMUNPU_OK;
  uint64_t a_stride =
      in->a_row_stride ? in->a_row_stride : (uint64_t)width * as;
  uint64_t c_stride =
      in->c_row_stride ? in->c_row_stride : (uint64_t)width * cs;

  uint8_t *A = NULL, *B = NULL, *C = NULL;
  emunpu_status_t st = emunpu_core_resolve(
      core, in->a_space, in->a_offset, g, gcap,
      (uint64_t)(table_rows - 1) * a_stride + (uint64_t)width * as, &A);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(core, in->b_space, in->b_offset, g, gcap,
                           (uint64_t)count * bs, &B);
  if (st != EMUNPU_OK) return st;
  st = emunpu_core_resolve(
      core, in->c_space, in->c_offset, g, gcap,
      (uint64_t)(count - 1) * c_stride + (uint64_t)width * cs, &C);
  if (st != EMUNPU_OK) return st;

  for (uint32_t i = 0; i < count; ++i) {
    double raw = emunpu_dtype_load(B, i, in->b_dtype);
    if (raw < 0.0 || raw >= (double)table_rows)
      return EMUNPU_ERROR_OUT_OF_RANGE;
    const uint8_t* src = A + (uint64_t)raw * a_stride;
    uint8_t* dst = C + i * c_stride;
    for (uint32_t j = 0; j < width; ++j) {
      emunpu_dtype_store(dst, j, in->c_dtype,
                         emunpu_dtype_load(src, j, in->a_dtype));
    }
  }
  return EMUNPU_OK;
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_core_execute(emunpu_core_t* core,
                                    const emunpu_instr_t* in,
                                    uint8_t* global_arena,
                                    uint64_t global_capacity, int trace) {
  if (!core || !in) return EMUNPU_ERROR_INVALID_ARGUMENT;
  // A unit cannot do more work in one issue than it is wide: check that before
  // anything else, so an over-large instruction is a clean error rather than an
  // unbounded amount of silently-executed work.
  emunpu_status_t caps = emunpu_instr_validate(in);
  if (caps != EMUNPU_OK) {
    if (trace) {
      fprintf(stderr, "[core%u] REJECTED %s: exceeds unit capability (%s)\n",
              core->core_id, emunpu_iop_name(in->op), emunpu_status_name(caps));
    }
    return caps;
  }
  if (trace) emunpu_core_trace(core, in);
  emunpu_viz_instr(core->core_id, in);
  core->instr_count++;
  switch (in->op) {
    case EMUNPU_I_DMA_G2L:
    case EMUNPU_I_DMA_L2G:
      return emunpu_core_dma(core, in, global_arena, global_capacity);
    case EMUNPU_I_MMA16:
      return emunpu_core_mma16(core, in, global_arena, global_capacity);
    case EMUNPU_I_MATVEC:
      return emunpu_core_matvec(core, in, global_arena, global_capacity);
    case EMUNPU_I_EW:
      return emunpu_core_ew(core, in, global_arena, global_capacity);
    case EMUNPU_I_REDUCE:
      return emunpu_core_reduce(core, in, global_arena, global_capacity);
    case EMUNPU_I_FMA:
      return emunpu_core_fma(core, in, global_arena, global_capacity);
    case EMUNPU_I_CONVERT:
      return emunpu_core_convert(core, in, global_arena, global_capacity);
    case EMUNPU_I_ACT:
      return emunpu_core_act(core, in, global_arena, global_capacity);
    case EMUNPU_I_SOFTMAX:
      return emunpu_core_softmax(core, in, global_arena, global_capacity);
    case EMUNPU_I_NORM:
      return emunpu_core_norm(core, in, global_arena, global_capacity);
    case EMUNPU_I_TRANSPOSE:
      return emunpu_core_transpose(core, in, global_arena, global_capacity);
    case EMUNPU_I_IM2COL:
      return emunpu_core_im2col(core, in, global_arena, global_capacity);
    case EMUNPU_I_POOL2D:
      return emunpu_core_pool2d(core, in, global_arena, global_capacity);
    case EMUNPU_I_GATHER:
      return emunpu_core_gather(core, in, global_arena, global_capacity);
    default:
      return EMUNPU_ERROR_UNIMPLEMENTED;
  }
}

emunpu_status_t emunpu_core_run(emunpu_core_t* core, uint8_t* global_arena,
                                uint64_t global_capacity, int trace) {
  if (!core) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_status_t st = EMUNPU_OK;
  for (uint32_t i = 0; i < core->queue_len; ++i) {
    st = emunpu_core_execute(core, &core->queue[i], global_arena,
                             global_capacity, trace);
    if (st != EMUNPU_OK) break;
  }
  emunpu_core_queue_reset(core);
  return st;
}
