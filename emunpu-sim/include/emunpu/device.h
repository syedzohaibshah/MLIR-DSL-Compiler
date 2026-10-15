// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Emulated multi-core NPU device + runtime API.
//
// This is the top-level "driver API" for the simulated accelerator, in the
// spirit of a CUDA/HIP runtime: it owns the centralized global memory, the
// array of IREE_EMUNPU_NUM_CORES compute cores, and a set of execution streams.
// It exposes:
//   * tensor allocation / free in global memory
//   * upload / download between host and global memory
//   * compute: matmul, gemm (+bias/+relu), matrix-vector, vector/elementwise,
//     reductions, fused multiply-add, datatype conversion
//   * ML operations: parameterized activations, linear layers with a fused
//     activation epilogue, softmax / log-softmax, layer & RMS normalization,
//     transpose, im2col, conv2d, 2D pooling, and gather (embedding lookup)
//   * streams (ordered async command lists), events, and synchronization
//     primitives (stream/event/device sync + a device-wide barrier)
//
// Execution is functionally synchronous and single-threaded: streams record
// ordered command lists that are drained on synchronize. This is enough to
// model the architecture (work fans out across the cores via the scheduler;
// see emunpu_scheduler.h) and produce bit-exact results, without a concurrency
// or timing model.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_DEVICE_H_
#define EMUNPU_DEVICE_H_

#include "emunpu/global_mem.h"
#include "emunpu/scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct emunpu_device_t emunpu_device_t;
typedef struct emunpu_stream_t emunpu_stream_t;
typedef struct emunpu_event_t emunpu_event_t;

//===----------------------------------------------------------------------===//
// Device lifecycle
//===----------------------------------------------------------------------===//

// Creates a device with |global_mem_bytes| of global memory (0 => default) and
// IREE_EMUNPU_NUM_CORES cores. Tracing follows IREE_EMUNPU_TRACE.
EMUNPU_API emunpu_status_t emunpu_device_create(uint64_t global_mem_bytes,
                                     emunpu_device_t** out_device);
EMUNPU_API void emunpu_device_destroy(emunpu_device_t* device);

// Number of compute cores in the device.
EMUNPU_API uint32_t emunpu_device_num_cores(const emunpu_device_t* device);

//===----------------------------------------------------------------------===//
// Hardware capabilities
//
// Everything a producer needs to plan work: how many cores there are, how much
// memory they can see, and how much work one instruction of each functional unit
// may specify. Query this instead of hard-coding the arch.h macros -- it is the
// same information a CUDA program gets from cudaGetDeviceProperties, and it is
// what makes the tiling in src/device.c (or in your own scheduler) portable to a
// differently-configured device.
//===----------------------------------------------------------------------===//

typedef struct emunpu_caps_t {
  uint32_t num_cores;         // cores online (see IREE_EMUNPU_NUM_CORES)
  uint64_t global_mem_bytes;  // shared arena capacity
  uint64_t local_mem_bytes;   // per-core scratchpad capacity
  uint32_t max_rank;          // tensor rank a descriptor can track
  uint32_t dtype_count;       // number of supported element types

  // Per-instruction limits of each functional unit. Exceeding one is an error,
  // not a slow path (see emunpu_instr_validate).
  uint32_t mma_dim;         // matrix engine: max M, N, K of one MMA/MATVEC
  uint32_t vector_lanes;    // vector ALU: max elements of one EW/FMA/CONVERT
  uint32_t sfu_lanes;       // special-function unit: max elements of one ACT
  uint32_t max_row_width;   // row unit: widest row REDUCE/SOFTMAX/NORM can take
  uint32_t transpose_dim;   // max edge of one TRANSPOSE block
  uint32_t gather_rows;     // max rows of one GATHER
  uint32_t im2col_rows;     // max patch rows of one IM2COL
  uint64_t dma_max_bytes;   // max bytes of one DMA descriptor
} emunpu_caps_t;

// Fills |out_caps| with |device|'s capabilities.
EMUNPU_API emunpu_status_t emunpu_device_query_caps(const emunpu_device_t* device,
                                                   emunpu_caps_t* out_caps);

//===----------------------------------------------------------------------===//
// Execution scope: choosing how many cores an operation uses
//
// By default every operation fans out across all online cores. An execution
// scope narrows that to the core range [first_core, first_core + core_count),
// which is how you dedicate part of the device to an operation, measure scaling,
// or partition the device between concurrent pipelines. It applies to every
// subsequent high-level compute call (including work recorded on a stream, which
// captures the scope in force at submission time).
//===----------------------------------------------------------------------===//

// Restricts subsequent operations to |core_count| cores starting at
// |first_core|. Returns INVALID_ARGUMENT if the range is empty or exceeds the
// online core count.
EMUNPU_API emunpu_status_t emunpu_device_set_exec_scope(emunpu_device_t* device,
                                                        uint32_t first_core,
                                                        uint32_t core_count);

// Reads the current scope (either pointer may be NULL).
EMUNPU_API void emunpu_device_get_exec_scope(const emunpu_device_t* device,
                                             uint32_t* out_first_core,
                                             uint32_t* out_core_count);

// Restores the default scope: all online cores.
EMUNPU_API void emunpu_device_reset_exec_scope(emunpu_device_t* device);

//===----------------------------------------------------------------------===//
// Low-level instruction programming
//
// The calls above are conveniences: each one tiles a tensor-sized operation into
// capability-sized instructions and deals them across the execution scope. This
// section is the layer underneath -- program the cores directly, the way a code
// generator or a hand-written kernel would:
//
//   emunpu_caps_t caps; emunpu_device_query_caps(dev, &caps);
//   emunpu_global_mem_offset(gm, tensor, &a_global);       // operand addresses
//   emunpu_device_local_alloc(dev, core, bytes, &a_local); // scratchpad
//   emunpu_device_enqueue(dev, core, &dma_in);             // build the stream
//   emunpu_device_enqueue(dev, core, &mma);
//   emunpu_device_enqueue(dev, core, &dma_out);
//   emunpu_device_launch(dev);                             // run every core
//
// You choose the tiling, the core assignment, and whether operands live in
// global memory or are staged through each core's scratchpad. Nothing is
// validated for you except the hardware's own limits (per-instruction capability
// and operand bounds), which is exactly what real hardware checks.
//===----------------------------------------------------------------------===//

// Bump-allocates |bytes| of |core_id|'s local scratchpad, returning the local
// byte offset to use in an instruction's *_offset with EMUNPU_SPACE_LOCAL.
EMUNPU_API emunpu_status_t emunpu_device_local_alloc(emunpu_device_t* device,
                                                     uint32_t core_id,
                                                     uint64_t bytes,
                                                     uint64_t* out_offset);

// Frees every local allocation on |core_id| at once (resets the bump pointer).
EMUNPU_API emunpu_status_t emunpu_device_local_reset(emunpu_device_t* device,
                                                     uint32_t core_id);

// Appends |instr| to |core_id|'s command queue after checking it against the
// unit capability limits, so a bad instruction is rejected at enqueue rather
// than mid-launch.
EMUNPU_API emunpu_status_t emunpu_device_enqueue(emunpu_device_t* device,
                                                 uint32_t core_id,
                                                 const emunpu_instr_t* instr);

// Executes every core's queued instructions (in order per core) and clears the
// queues. Returns the first failure; the remaining queues are dropped.
EMUNPU_API emunpu_status_t emunpu_device_launch(emunpu_device_t* device);

// Per-core execution counters, so a hand-written schedule can be measured and
// its load balance checked.
typedef struct emunpu_core_stats_t {
  uint64_t instr_count;  // instructions executed since device creation
  uint64_t mma_count;    // MMA issues
  uint64_t dma_bytes;    // bytes moved by DMA
  uint32_t queue_len;    // instructions currently queued (not yet launched)
  uint64_t local_used;   // scratchpad bytes currently allocated
} emunpu_core_stats_t;

EMUNPU_API emunpu_status_t emunpu_device_core_stats(const emunpu_device_t* device,
                                                    uint32_t core_id,
                                                    emunpu_core_stats_t* out);

// Direct access to the global memory subsystem (e.g. for FlatBuffer
// serialization of the layout).
EMUNPU_API emunpu_global_mem_t* emunpu_device_global_mem(emunpu_device_t* device);

//===----------------------------------------------------------------------===//
// Tensor allocation + host<->device transfer
//===----------------------------------------------------------------------===//

EMUNPU_API emunpu_status_t emunpu_device_alloc(emunpu_device_t* device,
                                    emunpu_dtype_t dtype, uint32_t rank,
                                    const uint32_t* shape, const char* name,
                                    emunpu_tensor_id_t* out_id);
EMUNPU_API emunpu_status_t emunpu_device_free(emunpu_device_t* device,
                                   emunpu_tensor_id_t id);

// Copies |byte_count| bytes from host |src| into tensor |id| (upload) or from
// the tensor into host |dst| (download). These model the host<->global DMA.
EMUNPU_API emunpu_status_t emunpu_device_upload(emunpu_device_t* device,
                                     emunpu_tensor_id_t id, const void* src,
                                     uint64_t byte_count);
EMUNPU_API emunpu_status_t emunpu_device_download(emunpu_device_t* device,
                                       emunpu_tensor_id_t id, void* dst,
                                       uint64_t byte_count);

//===----------------------------------------------------------------------===//
// Streams + events + synchronization
//===----------------------------------------------------------------------===//

// Creates/destroys an ordered command stream bound to |device|. Passing a NULL
// stream to a compute call executes it immediately (blocking, default stream).
EMUNPU_API emunpu_status_t emunpu_stream_create(emunpu_device_t* device,
                                     emunpu_stream_t** out_stream);
EMUNPU_API void emunpu_stream_destroy(emunpu_stream_t* stream);

// Drains all pending commands on |stream| (executes them in order).
EMUNPU_API emunpu_status_t emunpu_stream_synchronize(emunpu_stream_t* stream);

// Creates/destroys an event used to order work across streams.
EMUNPU_API emunpu_status_t emunpu_event_create(emunpu_device_t* device,
                                    emunpu_event_t** out_event);
EMUNPU_API void emunpu_event_destroy(emunpu_event_t* event);

// Records |event| at the current tail of |stream|'s command list.
EMUNPU_API emunpu_status_t emunpu_event_record(emunpu_event_t* event,
                                    emunpu_stream_t* stream);
// Blocks |stream| until |event| is reached (drains the event's stream so that
// subsequent |stream| work observes its results). NULL stream waits on the
// host.
EMUNPU_API emunpu_status_t emunpu_stream_wait_event(emunpu_stream_t* stream,
                                         emunpu_event_t* event);
// Ensures |event| has completed (drains its recording stream).
EMUNPU_API emunpu_status_t emunpu_event_synchronize(emunpu_event_t* event);

// Drains every stream on the device (host waits for all work to finish).
EMUNPU_API emunpu_status_t emunpu_device_synchronize(emunpu_device_t* device);
// Barrier: drains all streams and quiesces all cores. Equivalent to a full
// device synchronize; named distinctly to model an explicit cross-core barrier.
EMUNPU_API emunpu_status_t emunpu_device_barrier(emunpu_device_t* device);

//===----------------------------------------------------------------------===//
// Compute API
//
// Each call either executes immediately (stream == NULL) or is appended to the
// given stream and runs at synchronize. All operands are global-memory tensors.
//===----------------------------------------------------------------------===//

// C[M x N] = A[M x K] * B, B is K x N (transpose_b==0) or N x K
// (transpose_b==1). Tiled into 16x16 blocks and distributed across the cores by
// the scheduler (the runtime decides the tiling here).
EMUNPU_API emunpu_status_t emunpu_matmul(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              emunpu_tensor_id_t b, uint32_t M, uint32_t N,
                              uint32_t K, int transpose_b);

// Same matmul, but replays a precomputed (compiler-decided) tile schedule
// instead of tiling on-device. Used by the IREE HAL path, where tiling is a
// compile-time concern. Executes immediately (no stream); the caller stages the
// operands into global memory first, exactly as for emunpu_matmul.
EMUNPU_API emunpu_status_t emunpu_matmul_scheduled(emunpu_device_t* device,
                                        emunpu_tensor_id_t c,
                                        emunpu_tensor_id_t a,
                                        emunpu_tensor_id_t b, uint32_t M,
                                        uint32_t N, uint32_t K, int transpose_b,
                                        const emunpu_gemm_tile_t* tiles,
                                        uint32_t tile_count);

// GEMM with an optional row-bias add and fused ReLU epilogue:
//   C = relu?( A*B + bias )    (bias broadcast per output column; 0 to skip)
EMUNPU_API emunpu_status_t emunpu_gemm(emunpu_device_t* device, emunpu_stream_t* stream,
                            emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                            emunpu_tensor_id_t b, emunpu_tensor_id_t bias,
                            uint32_t M, uint32_t N, uint32_t K, int transpose_b,
                            int fuse_relu);

// y[M] = A[M x K] * x[K].
EMUNPU_API emunpu_status_t emunpu_matvec(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t y, emunpu_tensor_id_t a,
                              emunpu_tensor_id_t x, uint32_t M, uint32_t K);

// Elementwise vector op over |count| elements. |b| may be 0 for unary kinds
// (relu/copy). |bcast_period| broadcasts b (0 => equal-shape). Distributed
// across cores by splitting the element range.
EMUNPU_API emunpu_status_t emunpu_vector_op(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, emunpu_tensor_id_t b,
                                 uint32_t count, emunpu_ew_kind_t kind,
                                 uint32_t bcast_period);

// Reduces each of |rows| rows of width |width| to one output element.
EMUNPU_API emunpu_status_t emunpu_reduce(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              uint32_t rows, uint32_t width,
                              emunpu_reduce_kind_t kind);

// c[i] = a[i] * b[i] + c[i] over |count| elements.
EMUNPU_API emunpu_status_t emunpu_fma(emunpu_device_t* device, emunpu_stream_t* stream,
                           emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                           emunpu_tensor_id_t b, uint32_t count);

// c[i] = convert(a[i]); the dtypes come from the tensor descriptors.
EMUNPU_API emunpu_status_t emunpu_convert(emunpu_device_t* device, emunpu_stream_t* stream,
                               emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                               uint32_t count);

//===----------------------------------------------------------------------===//
// Machine-learning operations
//
// Everything below is built on the same cores and micro-ISA as the primitives
// above: activations run on the special-function unit, softmax/normalization on
// the vector unit, and convolution lowers to im2col + GEMM so it uses the very
// same 16x16 tensor cores. Work is fanned across the cores per op (rows,
// element ranges, or CHW planes, depending on the op).
//===----------------------------------------------------------------------===//

// A fused activation epilogue: the kind plus its scalar parameters. Passed by
// pointer so NULL means "no activation".
typedef struct emunpu_act_t {
  emunpu_act_kind_t kind;
  double alpha;  // LEAKY_RELU / ELU slope, CLAMP lower bound
  double beta;   // CLAMP upper bound
} emunpu_act_t;

// Geometry of a 2D sliding-window op (convolution / pooling) over an NCHW
// input. The output extents are derived, not stored -- see emunpu_window_output.
typedef struct emunpu_window_t {
  uint32_t batch;
  uint32_t channels;  // input channels
  uint32_t in_h, in_w;
  uint32_t kernel_h, kernel_w;
  uint32_t stride_h, stride_w;
  uint32_t pad_h, pad_w;
} emunpu_window_t;

// Output spatial extents of |window| (0 if the window does not fit the input).
EMUNPU_API void emunpu_window_output(const emunpu_window_t* window,
                                     uint32_t* out_h, uint32_t* out_w);

// c[i] = act(a[i]) over |count| elements. |alpha| / |beta| are the activation
// kind's parameters (ignored by kinds that take none).
EMUNPU_API emunpu_status_t emunpu_activation(emunpu_device_t* device,
                                  emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                  emunpu_tensor_id_t a, uint32_t count,
                                  emunpu_act_kind_t kind, double alpha,
                                  double beta);

// A full linear layer: C = act?( A*B (+ bias) ), i.e. emunpu_gemm generalized
// from ReLU-only to any activation. |bias| may be 0 and |act| may be NULL.
EMUNPU_API emunpu_status_t emunpu_linear(emunpu_device_t* device,
                              emunpu_stream_t* stream, emunpu_tensor_id_t c,
                              emunpu_tensor_id_t a, emunpu_tensor_id_t b,
                              emunpu_tensor_id_t bias, uint32_t M, uint32_t N,
                              uint32_t K, int transpose_b,
                              const emunpu_act_t* act);

// Row-wise softmax over |rows| rows of |width| elements (numerically stable).
// |log_softmax| emits log(softmax(x)) instead.
EMUNPU_API emunpu_status_t emunpu_softmax(emunpu_device_t* device,
                               emunpu_stream_t* stream, emunpu_tensor_id_t c,
                               emunpu_tensor_id_t a, uint32_t rows,
                               uint32_t width, int log_softmax);

// Layer normalization over the last dimension:
//   c = (a - mean(a)) / sqrt(var(a) + eps) * gamma + beta
// |gamma| / |beta| are length-|width| vectors broadcast across rows; either may
// be 0 to skip that term.
EMUNPU_API emunpu_status_t emunpu_layernorm(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, emunpu_tensor_id_t gamma,
                                 emunpu_tensor_id_t beta, uint32_t rows,
                                 uint32_t width, double eps);

// RMS normalization over the last dimension (no mean subtraction):
//   c = a / sqrt(mean(a^2) + eps) * gamma      (|gamma| may be 0)
EMUNPU_API emunpu_status_t emunpu_rmsnorm(emunpu_device_t* device,
                               emunpu_stream_t* stream, emunpu_tensor_id_t c,
                               emunpu_tensor_id_t a, emunpu_tensor_id_t gamma,
                               uint32_t rows, uint32_t width, double eps);

// c[cols x rows] = transpose(a[rows x cols]).
EMUNPU_API emunpu_status_t emunpu_transpose(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, uint32_t rows,
                                 uint32_t cols);

// 2D max/average pooling: a is NCHW [batch, channels, in_h, in_w], c is
// NCHW [batch, channels, out_h, out_w]. Padding taps are excluded from both the
// max and the average (PyTorch's count_include_pad=False).
EMUNPU_API emunpu_status_t emunpu_pool2d(emunpu_device_t* device,
                              emunpu_stream_t* stream, emunpu_tensor_id_t c,
                              emunpu_tensor_id_t a,
                              const emunpu_window_t* window,
                              emunpu_pool_kind_t kind);

// Convolution patch extraction: a is NCHW, c is the patch matrix
// [batch * out_h * out_w, channels * kernel_h * kernel_w] that turns a
// convolution into a GEMM. Exposed because it is useful on its own (and is what
// emunpu_conv2d issues first).
EMUNPU_API emunpu_status_t emunpu_im2col(emunpu_device_t* device,
                              emunpu_stream_t* stream, emunpu_tensor_id_t c,
                              emunpu_tensor_id_t a,
                              const emunpu_window_t* window);

// 2D convolution, lowered to im2col + GEMM (+ bias + activation + transpose):
//   input   NCHW [batch, channels, in_h, in_w]
//   weight  [out_channels, channels * kernel_h * kernel_w]
//           (torch.nn.Conv2d's [out_channels, channels, kh, kw] weight,
//            flattened -- the memory layout is identical)
//   bias    [out_channels] or 0
//   output  NCHW [batch, out_channels, out_h, out_w]
// Allocates (and releases) its own global-memory scratch for the patch matrix
// and the GEMM result.
EMUNPU_API emunpu_status_t emunpu_conv2d(emunpu_device_t* device,
                              emunpu_stream_t* stream, emunpu_tensor_id_t out,
                              emunpu_tensor_id_t in, emunpu_tensor_id_t weight,
                              emunpu_tensor_id_t bias,
                              const emunpu_window_t* window,
                              uint32_t out_channels, const emunpu_act_t* act);

// Row gather / embedding lookup: c[i, :] = table[indices[i], :] for |count|
// indices. |table_rows| bounds-checks every index; |indices| is normally an
// integer-dtype tensor.
EMUNPU_API emunpu_status_t emunpu_gather(emunpu_device_t* device,
                              emunpu_stream_t* stream, emunpu_tensor_id_t c,
                              emunpu_tensor_id_t table,
                              emunpu_tensor_id_t indices, uint32_t table_rows,
                              uint32_t count, uint32_t width);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_DEVICE_H_
