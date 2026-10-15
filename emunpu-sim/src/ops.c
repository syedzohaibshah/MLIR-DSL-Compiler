// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Sub-operation tables for the micro-ISA: mnemonics for every opcode/kind, and
// the activation unit's scalar transfer function.
//
// The transfer function lives here (rather than inline in the ACT kernel) for
// two reasons: the trace/visualizer need the same names the kernels dispatch
// on, and a host-side reference model can call emunpu_act_apply to reproduce
// the device bit-for-bit. Everything is plain double math -- the per-element
// dtype narrowing happens in emunpu_dtype_store, as with every other kernel.
//===----------------------------------------------------------------------===//

#include <math.h>

#include "emunpu/arch.h"

//===----------------------------------------------------------------------===//
// Mnemonics
//===----------------------------------------------------------------------===//

const char* emunpu_iop_name(emunpu_iopcode_t op) {
  switch (op) {
    case EMUNPU_I_DMA_G2L:
      return "DMA.g2l";
    case EMUNPU_I_DMA_L2G:
      return "DMA.l2g";
    case EMUNPU_I_MMA16:
      return "MMA16";
    case EMUNPU_I_MATVEC:
      return "MATVEC";
    case EMUNPU_I_EW:
      return "EW";
    case EMUNPU_I_REDUCE:
      return "REDUCE";
    case EMUNPU_I_FMA:
      return "FMA";
    case EMUNPU_I_CONVERT:
      return "CONVERT";
    case EMUNPU_I_ACT:
      return "ACT";
    case EMUNPU_I_SOFTMAX:
      return "SOFTMAX";
    case EMUNPU_I_NORM:
      return "NORM";
    case EMUNPU_I_TRANSPOSE:
      return "TRANSPOSE";
    case EMUNPU_I_IM2COL:
      return "IM2COL";
    case EMUNPU_I_POOL2D:
      return "POOL2D";
    case EMUNPU_I_GATHER:
      return "GATHER";
    default:
      return "????";
  }
}

int emunpu_ew_is_binary(emunpu_ew_kind_t kind) {
  switch (kind) {
    case EMUNPU_EW_ADD:
    case EMUNPU_EW_SUB:
    case EMUNPU_EW_MUL:
    case EMUNPU_EW_DIV:
    case EMUNPU_EW_MAX:
    case EMUNPU_EW_MIN:
    case EMUNPU_EW_POW:
      return 1;
    case EMUNPU_EW_RELU:
    case EMUNPU_EW_COPY:
    default:
      return 0;
  }
}

const char* emunpu_ew_name(emunpu_ew_kind_t kind) {
  switch (kind) {
    case EMUNPU_EW_ADD:
      return "add";
    case EMUNPU_EW_SUB:
      return "sub";
    case EMUNPU_EW_MUL:
      return "mul";
    case EMUNPU_EW_DIV:
      return "div";
    case EMUNPU_EW_MAX:
      return "max";
    case EMUNPU_EW_MIN:
      return "min";
    case EMUNPU_EW_RELU:
      return "relu";
    case EMUNPU_EW_COPY:
      return "copy";
    case EMUNPU_EW_POW:
      return "pow";
    default:
      return "?";
  }
}

const char* emunpu_reduce_name(emunpu_reduce_kind_t kind) {
  switch (kind) {
    case EMUNPU_REDUCE_SUM:
      return "sum";
    case EMUNPU_REDUCE_MAX:
      return "max";
    case EMUNPU_REDUCE_MIN:
      return "min";
    case EMUNPU_REDUCE_MEAN:
      return "mean";
    case EMUNPU_REDUCE_PROD:
      return "prod";
    case EMUNPU_REDUCE_ARGMAX:
      return "argmax";
    case EMUNPU_REDUCE_ARGMIN:
      return "argmin";
    default:
      return "?";
  }
}

const char* emunpu_act_name(emunpu_act_kind_t kind) {
  switch (kind) {
    case EMUNPU_ACT_RELU:
      return "relu";
    case EMUNPU_ACT_LEAKY_RELU:
      return "leaky_relu";
    case EMUNPU_ACT_RELU6:
      return "relu6";
    case EMUNPU_ACT_CLAMP:
      return "clamp";
    case EMUNPU_ACT_SIGMOID:
      return "sigmoid";
    case EMUNPU_ACT_TANH:
      return "tanh";
    case EMUNPU_ACT_GELU:
      return "gelu";
    case EMUNPU_ACT_GELU_TANH:
      return "gelu_tanh";
    case EMUNPU_ACT_SILU:
      return "silu";
    case EMUNPU_ACT_ELU:
      return "elu";
    case EMUNPU_ACT_SOFTPLUS:
      return "softplus";
    case EMUNPU_ACT_HARDSIGMOID:
      return "hardsigmoid";
    case EMUNPU_ACT_HARDSWISH:
      return "hardswish";
    case EMUNPU_ACT_MISH:
      return "mish";
    case EMUNPU_ACT_EXP:
      return "exp";
    case EMUNPU_ACT_LOG:
      return "log";
    case EMUNPU_ACT_SQRT:
      return "sqrt";
    case EMUNPU_ACT_RSQRT:
      return "rsqrt";
    case EMUNPU_ACT_ABS:
      return "abs";
    case EMUNPU_ACT_NEG:
      return "neg";
    case EMUNPU_ACT_RECIP:
      return "recip";
    case EMUNPU_ACT_SIGN:
      return "sign";
    case EMUNPU_ACT_SQUARE:
      return "square";
    case EMUNPU_ACT_ERF:
      return "erf";
    default:
      return "?";
  }
}

const char* emunpu_norm_name(emunpu_norm_kind_t kind) {
  switch (kind) {
    case EMUNPU_NORM_LAYER:
      return "layernorm";
    case EMUNPU_NORM_RMS:
      return "rmsnorm";
    default:
      return "?";
  }
}

const char* emunpu_pool_name(emunpu_pool_kind_t kind) {
  switch (kind) {
    case EMUNPU_POOL_MAX:
      return "max";
    case EMUNPU_POOL_AVG:
      return "avg";
    default:
      return "?";
  }
}

//===----------------------------------------------------------------------===//
// Capability limits
//
// One instruction = one issue of one functional unit, so the amount of work it
// may name is bounded by that unit's width. Rejecting an oversized instruction
// here (rather than quietly executing it) is what makes the tiling decision a
// real, visible obligation of the producer.
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_instr_validate(const emunpu_instr_t* instr) {
  if (!instr) return EMUNPU_ERROR_INVALID_ARGUMENT;
  const uint32_t mma = IREE_EMUNPU_MMA_DIM;
  switch (instr->op) {
    case EMUNPU_I_DMA_G2L:
    case EMUNPU_I_DMA_L2G: {
      size_t elem = emunpu_dtype_size(instr->c_dtype);
      if (!elem) return EMUNPU_ERROR_INVALID_ARGUMENT;
      uint64_t rows = instr->M ? instr->M : 1;
      if (rows * (uint64_t)instr->N * elem > IREE_EMUNPU_DMA_MAX_BYTES) {
        return EMUNPU_ERROR_OUT_OF_RANGE;
      }
      return EMUNPU_OK;
    }
    case EMUNPU_I_MMA16:
    case EMUNPU_I_MATVEC:
      // The matrix engine is a fixed MMA_DIM x MMA_DIM systolic tile.
      if (instr->M > mma || instr->N > mma || instr->K > mma) {
        return EMUNPU_ERROR_OUT_OF_RANGE;
      }
      return EMUNPU_OK;
    case EMUNPU_I_EW:
    case EMUNPU_I_FMA:
    case EMUNPU_I_CONVERT:
      if (instr->M > IREE_EMUNPU_VECTOR_LANES) return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    case EMUNPU_I_ACT:
      if (instr->M > IREE_EMUNPU_SFU_LANES) return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    case EMUNPU_I_REDUCE:
    case EMUNPU_I_SOFTMAX:
    case EMUNPU_I_NORM:
      // One row per issue: the unit has a single accumulator, and the row must
      // fit the row buffer it streams through.
      if (instr->M > 1) return EMUNPU_ERROR_OUT_OF_RANGE;
      if (instr->N > IREE_EMUNPU_MAX_ROW_WIDTH)
        return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    case EMUNPU_I_TRANSPOSE:
      // Transposes run through the matrix engine's tile shuffler.
      if (instr->M > mma || instr->N > mma) return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    case EMUNPU_I_IM2COL:
      // Produces at most one MMA operand tile's worth of patch rows.
      if (instr->M > mma) return EMUNPU_ERROR_OUT_OF_RANGE;
      if (instr->N > IREE_EMUNPU_MAX_ROW_WIDTH)
        return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    case EMUNPU_I_POOL2D: {
      // A row band of a single plane, bounded by the vector width it writes.
      uint32_t out_w = instr->spatial.out_w;
      if (out_w > IREE_EMUNPU_VECTOR_LANES) return EMUNPU_ERROR_OUT_OF_RANGE;
      if ((uint64_t)instr->M * out_w > IREE_EMUNPU_VECTOR_LANES) {
        return EMUNPU_ERROR_OUT_OF_RANGE;
      }
      return EMUNPU_OK;
    }
    case EMUNPU_I_GATHER:
      if (instr->M > mma) return EMUNPU_ERROR_OUT_OF_RANGE;
      if (instr->N > IREE_EMUNPU_MAX_ROW_WIDTH)
        return EMUNPU_ERROR_OUT_OF_RANGE;
      return EMUNPU_OK;
    default:
      return EMUNPU_ERROR_UNIMPLEMENTED;
  }
}

//===----------------------------------------------------------------------===//
// Activation transfer function
//===----------------------------------------------------------------------===//

static double emunpu_sigmoid(double x) {
  // Two-sided form so large |x| cannot overflow exp().
  if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
  double e = exp(x);
  return e / (1.0 + e);
}

// log(1 + exp(x)), saturating to the asymptotes instead of overflowing.
static double emunpu_softplus(double x) {
  if (x > 30.0) return x;
  if (x < -30.0) return exp(x);
  return log1p(exp(x));
}

static double emunpu_clamp_range(double x, double lo, double hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

double emunpu_act_apply(emunpu_act_kind_t kind, double x, double alpha,
                        double beta) {
  switch (kind) {
    case EMUNPU_ACT_RELU:
      return x > 0.0 ? x : 0.0;
    case EMUNPU_ACT_LEAKY_RELU:
      return x > 0.0 ? x : alpha * x;
    case EMUNPU_ACT_RELU6:
      return emunpu_clamp_range(x, 0.0, 6.0);
    case EMUNPU_ACT_CLAMP:
      return emunpu_clamp_range(x, alpha, beta);
    case EMUNPU_ACT_SIGMOID:
      return emunpu_sigmoid(x);
    case EMUNPU_ACT_TANH:
      return tanh(x);
    case EMUNPU_ACT_GELU:
      return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));  // x/sqrt(2)
    case EMUNPU_ACT_GELU_TANH:
      return 0.5 * x *
             (1.0 + tanh(0.79788456080286535588 *  // sqrt(2/pi)
                         (x + 0.044715 * x * x * x)));
    case EMUNPU_ACT_SILU:
      return x * emunpu_sigmoid(x);
    case EMUNPU_ACT_ELU:
      return x > 0.0 ? x : alpha * (exp(x) - 1.0);
    case EMUNPU_ACT_SOFTPLUS:
      return emunpu_softplus(x);
    case EMUNPU_ACT_HARDSIGMOID:
      return emunpu_clamp_range(x / 6.0 + 0.5, 0.0, 1.0);
    case EMUNPU_ACT_HARDSWISH:
      return x * emunpu_clamp_range(x / 6.0 + 0.5, 0.0, 1.0);
    case EMUNPU_ACT_MISH:
      return x * tanh(emunpu_softplus(x));
    case EMUNPU_ACT_EXP:
      return exp(x);
    case EMUNPU_ACT_LOG:
      return log(x);
    case EMUNPU_ACT_SQRT:
      return sqrt(x);
    case EMUNPU_ACT_RSQRT:
      return 1.0 / sqrt(x);
    case EMUNPU_ACT_ABS:
      return fabs(x);
    case EMUNPU_ACT_NEG:
      return -x;
    case EMUNPU_ACT_RECIP:
      return 1.0 / x;
    case EMUNPU_ACT_SIGN:
      return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    case EMUNPU_ACT_SQUARE:
      return x * x;
    case EMUNPU_ACT_ERF:
      return erf(x);
    default:
      return x;
  }
}
