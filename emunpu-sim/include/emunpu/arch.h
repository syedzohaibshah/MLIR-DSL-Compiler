// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Emulated multi-core NPU architecture description + low-level ISA.
//
// This header models a *realistic* coarse-grained NPU: a centralized global
// memory subsystem feeding an array of independent compute cores, each with its
// own local scratchpad memory, a tensor-core-style 16x16 MMA engine, a private
// command queue, and DMA access to global memory.
//
// Unlike the original whole-tensor `emunpu_isa.h` contract (which the IREE HAL
// loader still speaks), this is the *micro-architectural* layer that the device
// actually executes: a stream of tiled instructions (DMA / MMA16 / elementwise
// / reduction / ...) the on-device scheduler hands to individual cores. For
// matmul the *tiling decision* (16x16 blocking + per-core assignment) is now
// made by the compiler and shipped as a tile schedule in the executable blob
// (see emunpu_isa.h); the runtime replays it via emunpu_run_gemm_schedule. The
// device-native API path still tiles on-device (emunpu_schedule_gemm); both
// produce the same per-core instruction streams (see emunpu_scheduler.h).
//
// Beyond the matrix engine the device has a vector / special-function unit:
// elementwise arithmetic, reductions, parameterized activations, row-wise
// softmax and layer/RMS normalization, transpose, convolution patch extraction
// (im2col), 2D pooling, and row gather. docs/MICRO_ISA.md specifies every
// opcode's fields and exact semantics.
//
// It is deliberately free of any IREE dependency so the whole simulator builds
// and unit-tests standalone (plain C, only the C standard library). The IREE
// HAL driver wraps it; nothing here knows about iree_status_t / HAL plumbing.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_ARCH_H_
#define EMUNPU_ARCH_H_

#include <stddef.h>
#include <stdint.h>

#include "emunpu/export.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Architecture parameters
//===----------------------------------------------------------------------===//

// Number of independent compute cores in the emulated NPU.
#define IREE_EMUNPU_NUM_CORES 10

// The tensor-core MMA tile is MMA_DIM x MMA_DIM. Workloads larger than this are
// tiled into 16x16 blocks by the scheduler and accumulated across K-tiles.
#define IREE_EMUNPU_MMA_DIM 16

//===----------------------------------------------------------------------===//
// Per-instruction capability limits
//
// A functional unit processes a bounded amount of data per issue -- that bound
// IS the unit's width in silicon. The matrix engine has always modeled this
// (MMA16 rejects a tile wider than MMA_DIM); these constants extend the same
// honesty to the vector engine, the special-function unit, the row/reduction
// unit, and DMA.
//
// Consequences, which are the point of having them:
//   * An instruction that asks for more than its unit can do is *rejected*
//     (emunpu_instr_validate / EMUNPU_ERROR_OUT_OF_RANGE) -- the hardware never
//     silently does an unbounded amount of work.
//   * Tiling a tensor-sized operation into unit-sized instructions is therefore
//     the *producer's* job: a compiler, the device layer (src/device.c), or you,
//     via the low-level API in device.h. Query them at runtime with
//     emunpu_device_query_caps() rather than hard-coding these macros.
//   * The tile sizes are chosen so an instruction's working set fits
//     comfortably in a core's local scratchpad, so a producer can stage operands
//     through local memory (DMA in -> compute -> DMA out) if it wants to.
//===----------------------------------------------------------------------===//

// Vector-ALU width: elements one EW / FMA / CONVERT instruction may process.
#define IREE_EMUNPU_VECTOR_LANES 256

// Special-function-unit width: elements one ACT instruction may process. The SFU
// evaluates transcendentals, so like real hardware it is narrower than the
// vector ALU (here 1/4 of it).
#define IREE_EMUNPU_SFU_LANES 64

// The row unit (REDUCE / SOFTMAX / NORM) streams one row through a single
// accumulator per instruction, so an instruction covers exactly one row and the
// row must fit this buffer. Wider rows must be split by the caller.
#define IREE_EMUNPU_MAX_ROW_WIDTH 1024

// Largest transfer one DMA descriptor may move.
#define IREE_EMUNPU_DMA_MAX_BYTES (64u * 1024u)

// Per-core local scratchpad ("local memory") capacity in bytes. Sized to hold a
// handful of 16x16 tiles per operand plus accumulators with comfortable slack.
#define IREE_EMUNPU_LOCAL_MEM_BYTES (256 * 1024)

// Default centralized global memory capacity in bytes (configurable at device
// creation; this is the fallback when 0 is requested).
#define IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES (256ull * 1024ull * 1024ull)

// Fixed global-memory capacity of the emulated NPU *hardware*, in bytes. Unlike
// the device-native path (which may request any size, defaulting to 256MB), the
// HAL path models a fixed on-device global memory (like a real accelerator's
// on-chip SRAM / HBM partition): a stable 4MB resource that every workload's
// tensors share. Keeping it fixed — rather than sizing the arena per op — gives
// the visualizer's occupancy bar a stable denominator (e.g. "64.5 KB / 4.0 MB")
// instead of a per-op arena that always reads ~50% full. Grown only for a single
// op whose working set genuinely exceeds it (see emunpu_hw.c).
#define IREE_EMUNPU_GLOBAL_MEM_BYTES (4ull * 1024ull * 1024ull)

// Maximum tensor rank tracked by global-memory descriptors.
#define IREE_EMUNPU_MAX_RANK 4

//===----------------------------------------------------------------------===//
// Status codes
//
// The simulator is IREE-independent, so it reports errors via a small enum. The
// HAL glue (emunpu_hw.c) maps these onto iree_status_t.
//===----------------------------------------------------------------------===//

typedef enum emunpu_status_e {
  EMUNPU_OK = 0,
  EMUNPU_ERROR_INVALID_ARGUMENT,
  EMUNPU_ERROR_OUT_OF_MEMORY,
  EMUNPU_ERROR_OUT_OF_RANGE,
  EMUNPU_ERROR_UNIMPLEMENTED,
  EMUNPU_ERROR_FAILED_PRECONDITION,
  EMUNPU_ERROR_NOT_FOUND,
} emunpu_status_t;

EMUNPU_API const char* emunpu_status_name(emunpu_status_t status);

//===----------------------------------------------------------------------===//
// Datatypes
//
// The engine supports floating point (fp32/fp16/bf16) and integer (i32/i16/i8/
// u8) element types. Tensor-core MMA accumulates in a wider type than its
// inputs (fp32 for float inputs, i32 for integer inputs), so mixed precision
// (e.g. fp16 inputs / fp32 accumulate, or i8 inputs / i32 accumulate) and
// explicit datatype conversion fall out naturally from the per-element
// load/store accessors in emunpu_dtype.c.
//===----------------------------------------------------------------------===//

typedef enum emunpu_dtype_e {
  EMUNPU_DT_FP32 = 0,
  EMUNPU_DT_FP16 = 1,
  EMUNPU_DT_BF16 = 2,
  EMUNPU_DT_INT32 = 3,
  EMUNPU_DT_INT16 = 4,
  EMUNPU_DT_INT8 = 5,
  EMUNPU_DT_UINT8 = 6,
  EMUNPU_DT_COUNT,
} emunpu_dtype_t;

// Byte size of one element of |dtype| (0 for an invalid dtype).
EMUNPU_API size_t emunpu_dtype_size(emunpu_dtype_t dtype);

// Short human-readable name, e.g. "fp16" / "i8".
EMUNPU_API const char* emunpu_dtype_name(emunpu_dtype_t dtype);

// True if |dtype| is a floating point type (fp32/fp16/bf16).
EMUNPU_API int emunpu_dtype_is_float(emunpu_dtype_t dtype);

// True if |dtype| is a signed integer type (i32/i16/i8).
EMUNPU_API int emunpu_dtype_is_signed_int(emunpu_dtype_t dtype);

// Loads element |index| from |base| (typed by |dtype|) as a double, widening
// fp16/bf16/integers exactly. Store performs the reverse with the rounding /
// saturation appropriate to |dtype|. Routing every kernel through these makes
// the compute engine dtype-agnostic and mixed precision a no-op.
EMUNPU_API double emunpu_dtype_load(const void* base, size_t index, emunpu_dtype_t dtype);
EMUNPU_API void emunpu_dtype_store(void* base, size_t index, emunpu_dtype_t dtype,
                        double value);

// Scalar half-precision conversions, exposed for tests and the CONVERT op.
EMUNPU_API uint16_t emunpu_f32_to_fp16(float value);
EMUNPU_API float emunpu_fp16_to_f32(uint16_t bits);
EMUNPU_API uint16_t emunpu_f32_to_bf16(float value);
EMUNPU_API float emunpu_bf16_to_f32(uint16_t bits);

//===----------------------------------------------------------------------===//
// Micro-architectural ISA: per-core instructions
//
// A core executes a queue of these. Operands live in either global memory or
// the core's local scratchpad, addressed by a (memory space, byte offset) pair.
// The scheduler lowers a coarse op (e.g. a GEMM) into DMA + MMA16 (+ epilogue)
// instruction streams, one per assigned core.
//===----------------------------------------------------------------------===//

typedef enum emunpu_iopcode_e {
  EMUNPU_I_DMA_G2L = 1,  // copy a (possibly strided 2D) tile global -> local
  EMUNPU_I_DMA_L2G = 2,  // copy a (possibly strided 2D) tile local -> global
  EMUNPU_I_MMA16 = 3,    // C[MxN] += A[MxK] * B[KxN], dims <= MMA_DIM
  EMUNPU_I_MATVEC = 4,   // y[M] += A[MxK] * x[K]
  EMUNPU_I_EW = 5,       // elementwise unary/binary (see emunpu_ew_kind_t)
  EMUNPU_I_REDUCE =
      6,             // reduction over the last dim (see emunpu_reduce_kind_t)
  EMUNPU_I_FMA = 7,  // c[i] = a[i] * b[i] + c[i] (fused multiply-add)
  EMUNPU_I_CONVERT = 8,  // c[i] = convert(a[i]) across dtypes
  // --- Special-function / vector-engine extension (see docs/MICRO_ISA.md) ---
  EMUNPU_I_ACT = 9,       // c[i] = act(a[i]) (see emunpu_act_kind_t)
  EMUNPU_I_SOFTMAX = 10,  // row-wise (log-)softmax over the last dim
  EMUNPU_I_NORM = 11,     // row-wise layer/RMS normalization
  EMUNPU_I_TRANSPOSE = 12,  // c[NxM] = transpose(a[MxN])
  EMUNPU_I_IM2COL = 13,     // convolution patch extraction (CHW -> patch rows)
  EMUNPU_I_POOL2D = 14,     // 2D max/avg pooling over CHW planes
  EMUNPU_I_GATHER = 15,     // c[i,:] = a[idx[i],:] (embedding lookup)
} emunpu_iopcode_t;

// Short mnemonic for |op| ("MMA16", "ACT", ...), or "????" if unknown.
EMUNPU_API const char* emunpu_iop_name(emunpu_iopcode_t op);

// Memory space an operand address refers to.
typedef enum emunpu_space_e {
  EMUNPU_SPACE_GLOBAL = 0,  // offset is into the global memory arena
  EMUNPU_SPACE_LOCAL = 1,  // offset is into the issuing core's local scratchpad
} emunpu_space_t;

// Elementwise sub-operation (EMUNPU_I_EW). Unary kinds ignore operand b.
typedef enum emunpu_ew_kind_e {
  EMUNPU_EW_ADD = 0,
  EMUNPU_EW_SUB = 1,
  EMUNPU_EW_MUL = 2,
  EMUNPU_EW_DIV = 3,
  EMUNPU_EW_MAX = 4,
  EMUNPU_EW_MIN = 5,
  EMUNPU_EW_RELU = 6,  // unary: max(x, 0)
  EMUNPU_EW_COPY = 7,  // unary: x
  EMUNPU_EW_POW = 8,   // pow(a, b)
  EMUNPU_EW_COUNT,
} emunpu_ew_kind_t;

// True if |kind| consumes operand b (kinds not listed here are unary and the
// engine ignores b entirely). Ask this rather than comparing kind values: the
// binary and unary kinds are not contiguous.
EMUNPU_API int emunpu_ew_is_binary(emunpu_ew_kind_t kind);

// Short name for an elementwise kind, e.g. "add" / "relu".
EMUNPU_API const char* emunpu_ew_name(emunpu_ew_kind_t kind);

// Reduction sub-operation (EMUNPU_I_REDUCE). ARGMAX/ARGMIN produce the *index*
// of the extremum within the row, so their output tensor is normally an integer
// dtype.
typedef enum emunpu_reduce_kind_e {
  EMUNPU_REDUCE_SUM = 0,
  EMUNPU_REDUCE_MAX = 1,
  EMUNPU_REDUCE_MIN = 2,
  EMUNPU_REDUCE_MEAN = 3,
  EMUNPU_REDUCE_PROD = 4,
  EMUNPU_REDUCE_ARGMAX = 5,
  EMUNPU_REDUCE_ARGMIN = 6,
  EMUNPU_REDUCE_COUNT,
} emunpu_reduce_kind_t;

EMUNPU_API const char* emunpu_reduce_name(emunpu_reduce_kind_t kind);

//===----------------------------------------------------------------------===//
// Activation unit (EMUNPU_I_ACT)
//
// A real NPU pairs its matrix engine with a "special function unit" that
// applies a transcendental transfer function elementwise on the way out of the
// accumulator. This models that unit: one unary kind plus up to two scalar
// parameters (alpha/beta), applied through the same dtype accessors as every
// other kernel, so it works for any element type.
//
// Formulas match PyTorch's definitions (see docs/MICRO_ISA.md §5).
//===----------------------------------------------------------------------===//

typedef enum emunpu_act_kind_e {
  EMUNPU_ACT_RELU = 0,        // max(x, 0)
  EMUNPU_ACT_LEAKY_RELU = 1,  // x > 0 ? x : alpha * x
  EMUNPU_ACT_RELU6 = 2,       // min(max(x, 0), 6)
  EMUNPU_ACT_CLAMP = 3,       // min(max(x, alpha), beta)
  EMUNPU_ACT_SIGMOID = 4,     // 1 / (1 + exp(-x))
  EMUNPU_ACT_TANH = 5,        // tanh(x)
  EMUNPU_ACT_GELU = 6,        // x * 0.5 * (1 + erf(x / sqrt(2)))  (exact)
  EMUNPU_ACT_GELU_TANH = 7,   // the tanh approximation of GELU
  EMUNPU_ACT_SILU = 8,        // x * sigmoid(x)  (a.k.a. swish)
  EMUNPU_ACT_ELU = 9,         // x > 0 ? x : alpha * (exp(x) - 1)
  EMUNPU_ACT_SOFTPLUS = 10,   // log(1 + exp(x))
  EMUNPU_ACT_HARDSIGMOID = 11,  // clamp(x / 6 + 0.5, 0, 1)
  EMUNPU_ACT_HARDSWISH = 12,    // x * clamp(x / 6 + 0.5, 0, 1)
  EMUNPU_ACT_MISH = 13,         // x * tanh(softplus(x))
  EMUNPU_ACT_EXP = 14,          // exp(x)
  EMUNPU_ACT_LOG = 15,          // log(x)
  EMUNPU_ACT_SQRT = 16,         // sqrt(x)
  EMUNPU_ACT_RSQRT = 17,        // 1 / sqrt(x)
  EMUNPU_ACT_ABS = 18,          // |x|
  EMUNPU_ACT_NEG = 19,          // -x
  EMUNPU_ACT_RECIP = 20,        // 1 / x
  EMUNPU_ACT_SIGN = 21,         // -1 / 0 / +1
  EMUNPU_ACT_SQUARE = 22,       // x * x
  EMUNPU_ACT_ERF = 23,          // erf(x)
  EMUNPU_ACT_COUNT,
} emunpu_act_kind_t;

// The activation unit's scalar transfer function: what EMUNPU_I_ACT applies to
// every element. |alpha| / |beta| are the kind's parameters (unused kinds ignore
// them). Exposed so a host-side model can mirror the device exactly.
EMUNPU_API double emunpu_act_apply(emunpu_act_kind_t kind, double x, double alpha,
                                   double beta);

// Short name for an activation kind, e.g. "gelu" / "leaky_relu".
EMUNPU_API const char* emunpu_act_name(emunpu_act_kind_t kind);

// Row-wise normalization sub-operation (EMUNPU_I_NORM), with epsilon in param0.
typedef enum emunpu_norm_kind_e {
  // (x - mean(x)) / sqrt(var(x) + eps), var computed over the row (biased).
  EMUNPU_NORM_LAYER = 0,
  // x / sqrt(mean(x^2) + eps) -- no mean subtraction (LLaMA-style RMSNorm).
  EMUNPU_NORM_RMS = 1,
  EMUNPU_NORM_COUNT,
} emunpu_norm_kind_t;

EMUNPU_API const char* emunpu_norm_name(emunpu_norm_kind_t kind);

// Pooling sub-operation (EMUNPU_I_POOL2D). AVG averages only the in-bounds
// elements of a window (PyTorch's count_include_pad=False); MAX ignores
// out-of-bounds elements entirely.
typedef enum emunpu_pool_kind_e {
  EMUNPU_POOL_MAX = 0,
  EMUNPU_POOL_AVG = 1,
  EMUNPU_POOL_COUNT,
} emunpu_pool_kind_t;

EMUNPU_API const char* emunpu_pool_name(emunpu_pool_kind_t kind);

// Softmax flags (EMUNPU_I_SOFTMAX).
enum {
  // Emit log(softmax(x)) instead of softmax(x), computed in the numerically
  // stable form x - max - log(sum(exp(x - max))).
  EMUNPU_SOFTMAX_LOG = 1,
};

// MMA layout flags (matches the high-level ISA's transpose-B semantics).
enum {
  EMUNPU_MMA_STD = 0,          // B is K x N
  EMUNPU_MMA_TRANSPOSE_B = 1,  // B is N x K (nn.Linear's x * Wᵀ)
  // When set, MMA16 accumulates into C's current contents (used for the
  // 2nd..nth K-tile of a tiled GEMM). When clear, C is overwritten (the first
  // K-tile).
  EMUNPU_MMA_ACCUMULATE = 2,
};

// Spatial (convolution / pooling) parameters, used by EMUNPU_I_IM2COL and
// EMUNPU_I_POOL2D. Kept in a named sub-struct so the instruction stays flat and
// zero-initializable, and so a reader can tell at a glance which ops need it.
//
// Planes are CHW-contiguous: plane c of an image starts at c * in_h * in_w
// elements. |out_h| / |out_w| are the *derived* output extents
// (floor((in + 2*pad - kernel) / stride) + 1); the producer computes them once
// (emunpu_spatial_out_extent) so every core agrees on the same grid.
typedef struct emunpu_spatial_t {
  uint32_t in_h, in_w;
  // Channels per image (IM2COL only; a POOL2D instruction covers one plane,
  // selected by its operand offsets).
  uint32_t channels;
  uint32_t kernel_h, kernel_w;
  uint32_t stride_h, stride_w;
  uint32_t pad_h, pad_w;
  uint32_t out_h, out_w;
  // Index of the first output row this instruction emits: the first patch row
  // (IM2COL) or the first output row of the plane (POOL2D). The kernel needs the
  // *logical* row index -- not just the write offset -- to recover the window
  // position, which is what lets a row range be split across cores.
  uint32_t first_out_row;
} emunpu_spatial_t;

// Output extent of a convolution/pooling dimension:
//   floor((in + 2 * pad - kernel) / stride) + 1, or 0 if the window does not
// fit at all (an invalid configuration the caller should reject).
static inline uint32_t emunpu_spatial_out_extent(uint32_t in, uint32_t kernel,
                                                 uint32_t stride,
                                                 uint32_t pad) {
  if (stride == 0 || kernel == 0) return 0;
  uint32_t padded = in + 2 * pad;
  if (padded < kernel) return 0;
  return (padded - kernel) / stride + 1;
}

// A single core instruction. Not all fields are used by every opcode; unused
// fields are zero. Dimensions are element counts; offsets and strides are in
// bytes so a tile can be DMA'd out of a row-major matrix without copies.
typedef struct emunpu_instr_t {
  emunpu_iopcode_t op;

  // Element types of the two inputs, the accumulator, and the output. For most
  // ops these are equal; MMA/CONVERT use the spread for mixed precision.
  emunpu_dtype_t a_dtype;
  emunpu_dtype_t b_dtype;
  emunpu_dtype_t acc_dtype;
  emunpu_dtype_t c_dtype;

  // Operand addresses: (space, byte offset). b is unused by unary ops.
  emunpu_space_t a_space;
  emunpu_space_t b_space;
  emunpu_space_t c_space;
  uint64_t a_offset;
  uint64_t b_offset;
  uint64_t c_offset;

  // Problem dimensions. Interpretation per opcode:
  //   DMA_*  : M rows of N elements; *_row_stride give the source/dest strides.
  //   MMA16  : C[M x N] += A[M x K] * B[K x N]   (dims <= MMA_DIM)
  //   MATVEC : y[M] += A[M x K] * x[K]
  //   EW/FMA/CONVERT : M = element count (<= VECTOR_LANES).
  //   ACT : M = element count (<= SFU_LANES).
  //   REDUCE/SOFTMAX/NORM : M = 1 (one row per instruction),
  //                         N = row width (<= MAX_ROW_WIDTH).
  //   TRANSPOSE : reads A[M x N], writes C[N x M] (dims <= MMA_DIM).
  //   IM2COL : M = patch rows to emit, N = row width (channels*kh*kw).
  //   POOL2D : M = output rows of ONE plane, starting at spatial.first_out_row.
  //   GATHER : M = indices to gather, N = row width, K = table row count.
  // See emunpu_instr_validate() for the exact per-opcode bound on each field.
  uint32_t M;
  uint32_t N;
  uint32_t K;

  // Row strides in bytes for the A and C operands (DMA 2D copies and tiled MMA
  // reads/writes of a sub-block of a larger row-major matrix). 0 => contiguous
  // (row_stride == N * elem_size).
  uint32_t a_row_stride;
  uint32_t b_row_stride;
  uint32_t c_row_stride;

  // Opcode-specific modifier:
  //   MMA16   : bitmask of EMUNPU_MMA_* flags.
  //   EW      : emunpu_ew_kind_t.
  //   REDUCE  : emunpu_reduce_kind_t.
  //   ACT     : emunpu_act_kind_t.
  //   NORM    : emunpu_norm_kind_t.
  //   POOL2D  : emunpu_pool_kind_t.
  //   SOFTMAX : bitmask of EMUNPU_SOFTMAX_* flags.
  uint32_t flags;

  // For EW: broadcast period of operand b (0 => equal-shape; p>0 => b[i % p],
  // the row-bias pattern). Ignored by other ops.
  uint32_t bcast_period;

  // For EW with bcast_period != 0: the global element index this instruction's
  // slice starts at, so operand b is read as b[(bcast_phase + i) % period].
  // Without it, an instruction covering a slice of a larger elementwise op could
  // only be issued on a period boundary -- which a VECTOR_LANES-sized tile of a
  // wider-than-VECTOR_LANES broadcast period can never be.
  uint32_t bcast_phase;

  // Scalar operands of the flags-selected sub-operation:
  //   ACT  : param0 = alpha, param1 = beta (see emunpu_act_kind_t).
  //   NORM : param0 = epsilon.
  // Zero for every other opcode.
  double param0;
  double param1;

  // Window geometry for IM2COL / POOL2D. Zero for every other opcode.
  emunpu_spatial_t spatial;
} emunpu_instr_t;

// Checks |instr| against the issuing unit's per-instruction capability limits
// (§"Per-instruction capability limits" above): the element/row/tile counts a
// single instruction of this opcode may specify, and the DMA descriptor size.
// Returns EMUNPU_ERROR_OUT_OF_RANGE when the instruction asks for more work than
// the unit can do in one issue, or EMUNPU_ERROR_INVALID_ARGUMENT for a malformed
// opcode/dtype. Operand *addresses* are validated separately, at execution, by
// the core.
//
// The core calls this on every instruction before executing it, so a producer
// cannot exceed the hardware by accident; call it yourself to check a schedule
// without running it.
EMUNPU_API emunpu_status_t emunpu_instr_validate(const emunpu_instr_t* instr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_ARCH_H_
