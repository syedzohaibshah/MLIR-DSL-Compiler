// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emunpu/device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emunpu/viz.h"

//===----------------------------------------------------------------------===//
// Internal command list
//
// Streams record a list of primitive device commands; a synchronize drains the
// list by executing each command (fanning work across the cores). A "GEMM with
// bias/relu" decomposes into several primitives at submission time.
//===----------------------------------------------------------------------===//

typedef enum {
  EMUNPU_CMD_MATMUL,
  EMUNPU_CMD_MATVEC,
  EMUNPU_CMD_EW,
  EMUNPU_CMD_REDUCE,
  EMUNPU_CMD_FMA,
  EMUNPU_CMD_CONVERT,
  EMUNPU_CMD_ACT,
  EMUNPU_CMD_SOFTMAX,
  EMUNPU_CMD_NORM,
  EMUNPU_CMD_TRANSPOSE,
  EMUNPU_CMD_IM2COL,
  EMUNPU_CMD_POOL2D,
  EMUNPU_CMD_GATHER,
  // Releases a scratch tensor once the commands before it have executed. Lets a
  // composite op (conv2d) allocate intermediates at submission time and still
  // free them at the right point of a deferred stream.
  EMUNPU_CMD_FREE,
} emunpu_cmd_kind_t;

typedef struct emunpu_command_t {
  emunpu_cmd_kind_t kind;
  emunpu_tensor_id_t a, b, c;
  uint32_t M, N, K;
  uint32_t modifier;      // transpose_b / ew|reduce|act|norm|pool kind / flags
  uint32_t bcast_period;  // EW broadcast period
  // Byte offsets added to each operand's resolved base, so one command can
  // address a slice of a tensor (e.g. a single image of an NCHW batch).
  uint64_t a_byte_offset, b_byte_offset, c_byte_offset;
  double param0, param1;     // ACT alpha/beta, NORM epsilon
  emunpu_spatial_t spatial;  // IM2COL / POOL2D window geometry
  // Cores this command may use, captured from the device's execution scope when
  // the command was submitted (so a stream replays the scope that was in force
  // when the work was recorded, not whatever is set at synchronize time).
  uint32_t scope_first, scope_count;
} emunpu_command_t;

struct emunpu_stream_t {
  emunpu_device_t* device;
  emunpu_command_t* cmds;
  uint32_t len;
  uint32_t cap;
  uint32_t executed;  // number of commands already drained
};

struct emunpu_event_t {
  emunpu_device_t* device;
  emunpu_stream_t* stream;  // stream the event was recorded on (NULL if none)
  uint32_t recorded_pos;    // command index at record time
};

struct emunpu_device_t {
  emunpu_global_mem_t gm;
  emunpu_core_t cores[IREE_EMUNPU_NUM_CORES];
  uint32_t num_cores;
  int trace;

  // Execution scope: the core range high-level ops fan out across. Defaults to
  // every online core; emunpu_device_set_exec_scope narrows it.
  uint32_t scope_first;
  uint32_t scope_count;

  // Registry of live streams (for device_synchronize / barrier).
  emunpu_stream_t** streams;
  uint32_t stream_count;
  uint32_t stream_cap;
};

static int emunpu_trace_enabled(void) {
  const char* v = getenv("IREE_EMUNPU_TRACE");
  return (v && v[0] != '\0' && v[0] != '0') ? 1 : 0;
}

// Number of cores to bring online, from IREE_EMUNPU_NUM_CORES (clamped to
// [1, IREE_EMUNPU_NUM_CORES]). This lets a single binary run on any core count
// -- the compiler always bakes a 10-core round-robin schedule, and the replay
// path (emunpu_run_gemm_schedule) folds those core ids onto the cores that are
// actually online. Unset / invalid keeps all physical cores.
static uint32_t emunpu_env_num_cores(void) {
  const char* v = getenv("IREE_EMUNPU_NUM_CORES");
  if (!v || v[0] == '\0') return IREE_EMUNPU_NUM_CORES;
  long n = strtol(v, NULL, 10);
  if (n < 1) n = 1;
  if (n > IREE_EMUNPU_NUM_CORES) n = IREE_EMUNPU_NUM_CORES;
  return (uint32_t)n;
}

// Drains every core's queue (defined with the other execution helpers below).
static emunpu_status_t emunpu_run_all_cores(emunpu_device_t* device);

//===----------------------------------------------------------------------===//
// Device lifecycle
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_device_create(uint64_t global_mem_bytes,
                                     emunpu_device_t** out_device) {
  if (!out_device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  *out_device = NULL;
  emunpu_device_t* device = (emunpu_device_t*)calloc(1, sizeof(*device));
  if (!device) return EMUNPU_ERROR_OUT_OF_MEMORY;

  emunpu_status_t st =
      emunpu_global_mem_initialize(&device->gm, global_mem_bytes);
  if (st != EMUNPU_OK) {
    free(device);
    return st;
  }
  device->num_cores = emunpu_env_num_cores();
  for (uint32_t i = 0; i < device->num_cores; ++i) {
    st = emunpu_core_initialize(&device->cores[i], i);
    if (st != EMUNPU_OK) {
      for (uint32_t j = 0; j < i; ++j)
        emunpu_core_deinitialize(&device->cores[j]);
      emunpu_global_mem_deinitialize(&device->gm);
      free(device);
      return st;
    }
  }
  device->trace = emunpu_trace_enabled();
  device->scope_first = 0;
  device->scope_count = device->num_cores;
  emunpu_viz_run_begin(device->num_cores, device->gm.capacity);
  *out_device = device;
  return EMUNPU_OK;
}

void emunpu_device_destroy(emunpu_device_t* device) {
  if (!device) return;
  for (uint32_t i = 0; i < device->num_cores; ++i) {
    emunpu_core_deinitialize(&device->cores[i]);
  }
  emunpu_global_mem_deinitialize(&device->gm);
  free(device->streams);
  free(device);
}

uint32_t emunpu_device_num_cores(const emunpu_device_t* device) {
  return device ? device->num_cores : 0;
}

emunpu_global_mem_t* emunpu_device_global_mem(emunpu_device_t* device) {
  return device ? &device->gm : NULL;
}

//===----------------------------------------------------------------------===//
// Capabilities + execution scope
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_device_query_caps(const emunpu_device_t* device,
                                         emunpu_caps_t* out_caps) {
  if (!device || !out_caps) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_caps_t caps = {0};
  caps.num_cores = device->num_cores;
  caps.global_mem_bytes = device->gm.capacity;
  caps.local_mem_bytes = IREE_EMUNPU_LOCAL_MEM_BYTES;
  caps.max_rank = IREE_EMUNPU_MAX_RANK;
  caps.dtype_count = EMUNPU_DT_COUNT;
  caps.mma_dim = IREE_EMUNPU_MMA_DIM;
  caps.vector_lanes = IREE_EMUNPU_VECTOR_LANES;
  caps.sfu_lanes = IREE_EMUNPU_SFU_LANES;
  caps.max_row_width = IREE_EMUNPU_MAX_ROW_WIDTH;
  // Transposes, gathers and im2col all feed (or are fed by) the matrix engine,
  // so they inherit its tile height rather than having a width of their own.
  caps.transpose_dim = IREE_EMUNPU_MMA_DIM;
  caps.gather_rows = IREE_EMUNPU_MMA_DIM;
  caps.im2col_rows = IREE_EMUNPU_MMA_DIM;
  caps.dma_max_bytes = IREE_EMUNPU_DMA_MAX_BYTES;
  *out_caps = caps;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_device_set_exec_scope(emunpu_device_t* device,
                                             uint32_t first_core,
                                             uint32_t core_count) {
  if (!device || core_count == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if ((uint64_t)first_core + core_count > device->num_cores) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  device->scope_first = first_core;
  device->scope_count = core_count;
  return EMUNPU_OK;
}

void emunpu_device_get_exec_scope(const emunpu_device_t* device,
                                  uint32_t* out_first_core,
                                  uint32_t* out_core_count) {
  if (out_first_core) *out_first_core = device ? device->scope_first : 0;
  if (out_core_count) *out_core_count = device ? device->scope_count : 0;
}

void emunpu_device_reset_exec_scope(emunpu_device_t* device) {
  if (!device) return;
  device->scope_first = 0;
  device->scope_count = device->num_cores;
}

//===----------------------------------------------------------------------===//
// Low-level instruction programming
//===----------------------------------------------------------------------===//

static emunpu_core_t* emunpu_device_core(emunpu_device_t* device,
                                         uint32_t core_id) {
  if (!device || core_id >= device->num_cores) return NULL;
  return &device->cores[core_id];
}

emunpu_status_t emunpu_device_local_alloc(emunpu_device_t* device,
                                          uint32_t core_id, uint64_t bytes,
                                          uint64_t* out_offset) {
  emunpu_core_t* core = emunpu_device_core(device, core_id);
  if (!core || !out_offset) return EMUNPU_ERROR_INVALID_ARGUMENT;
  return emunpu_core_local_alloc(core, bytes, out_offset);
}

emunpu_status_t emunpu_device_local_reset(emunpu_device_t* device,
                                          uint32_t core_id) {
  emunpu_core_t* core = emunpu_device_core(device, core_id);
  if (!core) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_core_local_reset(core);
  return EMUNPU_OK;
}

emunpu_status_t emunpu_device_enqueue(emunpu_device_t* device, uint32_t core_id,
                                      const emunpu_instr_t* instr) {
  emunpu_core_t* core = emunpu_device_core(device, core_id);
  if (!core || !instr) return EMUNPU_ERROR_INVALID_ARGUMENT;
  // Fail at enqueue rather than mid-launch: a schedule that exceeds a unit's
  // capability is a bug in the producer, and this is where it can still be
  // fixed.
  emunpu_status_t st = emunpu_instr_validate(instr);
  if (st != EMUNPU_OK) return st;
  return emunpu_core_enqueue(core, instr);
}

emunpu_status_t emunpu_device_launch(emunpu_device_t* device) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  return emunpu_run_all_cores(device);
}

emunpu_status_t emunpu_device_core_stats(const emunpu_device_t* device,
                                         uint32_t core_id,
                                         emunpu_core_stats_t* out) {
  if (!device || !out || core_id >= device->num_cores) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  const emunpu_core_t* core = &device->cores[core_id];
  emunpu_core_stats_t stats = {0};
  stats.instr_count = core->instr_count;
  stats.mma_count = core->mma_count;
  stats.dma_bytes = core->dma_bytes;
  stats.queue_len = core->queue_len;
  stats.local_used = core->local_watermark;
  *out = stats;
  return EMUNPU_OK;
}

//===----------------------------------------------------------------------===//
// Tensor allocation + transfer
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_device_alloc(emunpu_device_t* device,
                                    emunpu_dtype_t dtype, uint32_t rank,
                                    const uint32_t* shape, const char* name,
                                    emunpu_tensor_id_t* out_id) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  return emunpu_global_mem_alloc(&device->gm, dtype, rank, shape, name, out_id);
}

emunpu_status_t emunpu_device_free(emunpu_device_t* device,
                                   emunpu_tensor_id_t id) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  return emunpu_global_mem_free(&device->gm, id);
}

emunpu_status_t emunpu_device_upload(emunpu_device_t* device,
                                     emunpu_tensor_id_t id, const void* src,
                                     uint64_t byte_count) {
  if (!device || !src) return EMUNPU_ERROR_INVALID_ARGUMENT;
  const emunpu_tensor_desc_t* d = emunpu_global_mem_lookup(&device->gm, id);
  if (!d) return EMUNPU_ERROR_NOT_FOUND;
  if (byte_count > d->size) return EMUNPU_ERROR_OUT_OF_RANGE;
  memcpy(emunpu_global_mem_host_ptr(&device->gm, id), src, byte_count);
  return EMUNPU_OK;
}

emunpu_status_t emunpu_device_download(emunpu_device_t* device,
                                       emunpu_tensor_id_t id, void* dst,
                                       uint64_t byte_count) {
  if (!device || !dst) return EMUNPU_ERROR_INVALID_ARGUMENT;
  const emunpu_tensor_desc_t* d = emunpu_global_mem_lookup(&device->gm, id);
  if (!d) return EMUNPU_ERROR_NOT_FOUND;
  if (byte_count > d->size) return EMUNPU_ERROR_OUT_OF_RANGE;
  memcpy(dst, emunpu_global_mem_host_ptr(&device->gm, id), byte_count);
  return EMUNPU_OK;
}

//===----------------------------------------------------------------------===//
// Core execution helpers
//===----------------------------------------------------------------------===//

// Runs every core's queued instructions against the global arena, reporting the
// first failure. Once a core has failed the remaining cores' queues are dropped
// rather than executed: the op is already broken, and leaving queued work
// behind would let it fire during the *next* op, against unrelated operands.
static emunpu_status_t emunpu_run_all_cores(emunpu_device_t* device) {
  emunpu_status_t result = EMUNPU_OK;
  for (uint32_t i = 0; i < device->num_cores; ++i) {
    if (result == EMUNPU_OK) {
      result = emunpu_core_run(&device->cores[i], device->gm.arena,
                               device->gm.capacity, device->trace);
    } else {
      emunpu_core_queue_reset(&device->cores[i]);
    }
  }
  emunpu_viz_mem(device->gm.bytes_in_use, device->gm.capacity);
  return result;
}

static emunpu_status_t emunpu_resolve(emunpu_device_t* device,
                                      emunpu_tensor_id_t id, uint64_t* offset,
                                      emunpu_dtype_t* dtype) {
  const emunpu_tensor_desc_t* d = emunpu_global_mem_lookup(&device->gm, id);
  if (!d) return EMUNPU_ERROR_NOT_FOUND;
  *offset = d->offset;
  if (dtype) *dtype = d->dtype;
  return EMUNPU_OK;
}

//===----------------------------------------------------------------------===//
// Issue helper: tile to the unit limits, then deal across the execution scope
//
// Every operation below has the same shape, which is the point of the
// capability model: split the work into instructions no larger than the issuing
// unit can execute in one go (emunpu_caps_t / emunpu_instr_validate), then deal
// those instructions across the cores of the command's execution scope. A chain
// of instructions that accumulates into the same output (a K-tile sequence)
// must stay on one core, in order -- that is what emunpu_issue_to() is for.
//===----------------------------------------------------------------------===//

typedef struct emunpu_issue_t {
  emunpu_device_t* device;
  uint32_t first_core;
  uint32_t core_count;
  uint32_t next;  // rotating slot within the scope
} emunpu_issue_t;

static void emunpu_issue_begin(emunpu_issue_t* it, emunpu_device_t* device,
                               const emunpu_command_t* cmd) {
  it->device = device;
  it->first_core = cmd->scope_first;
  it->core_count = cmd->scope_count ? cmd->scope_count : 1;
  it->next = 0;
}

// Issues |in| to a specific slot of the scope. Instructions sharing a slot run
// in issue order on one core, which is what an accumulation chain needs.
static emunpu_status_t emunpu_issue_to(emunpu_issue_t* it, uint32_t slot,
                                       const emunpu_instr_t* in) {
  uint32_t core = it->first_core + (slot % it->core_count);
  return emunpu_core_enqueue(&it->device->cores[core], in);
}

// Issues |in| to the next core, round-robin.
static emunpu_status_t emunpu_issue(emunpu_issue_t* it,
                                    const emunpu_instr_t* in) {
  return emunpu_issue_to(it, it->next++, in);
}

//===----------------------------------------------------------------------===//
// Primitive command execution
//===----------------------------------------------------------------------===//

static emunpu_status_t emunpu_exec_matmul(emunpu_device_t* device,
                                          const emunpu_command_t* cmd) {
  emunpu_gemm_desc_t g = {0};
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &g.a_offset, &g.a_dtype)) ||
      (st = emunpu_resolve(device, cmd->b, &g.b_offset, &g.b_dtype)) ||
      (st = emunpu_resolve(device, cmd->c, &g.c_offset, &g.c_dtype))) {
    return st;
  }
  g.a_offset += cmd->a_byte_offset;
  g.b_offset += cmd->b_byte_offset;
  g.c_offset += cmd->c_byte_offset;
  g.M = cmd->M;
  g.N = cmd->N;
  g.K = cmd->K;
  g.transpose_b = (int)cmd->modifier;

  // The scheduler already blocks the GEMM into MMA_DIM tiles; point it at the
  // scope's cores so a narrowed scope is honored.
  uint32_t tiles = 0;
  st = emunpu_schedule_gemm(device->cores + cmd->scope_first, cmd->scope_count,
                            &g, &tiles);
  if (st != EMUNPU_OK) return st;
  emunpu_viz_gemm(g.M, g.N, g.K, g.transpose_b, tiles, cmd->scope_count);
  if (device->trace) {
    fprintf(stderr, "[npu] GEMM %ux%ux%u %s-> %u tiles across %u cores\n", g.M,
            g.N, g.K, g.transpose_b ? "transpose_b " : "", tiles,
            cmd->scope_count);
  }
  return emunpu_run_all_cores(device);
}

// Elementwise: VECTOR_LANES elements per instruction.
//
// A broadcast operand (period != 0) is indexed by the *global* element index,
// so each instruction carries the phase its slice starts at. That is what lets
// a lane-sized tile sit anywhere inside a broadcast pattern -- including one
// whose period is wider than the vector unit.
static emunpu_status_t emunpu_exec_ew(emunpu_device_t* device,
                                      const emunpu_command_t* cmd) {
  emunpu_ew_kind_t kind = (emunpu_ew_kind_t)cmd->modifier;
  int binary = emunpu_ew_is_binary(kind);
  uint64_t a_off, b_off = 0, c_off;
  emunpu_dtype_t a_dt, b_dt = EMUNPU_DT_FP32, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  if (binary) {
    if ((st = emunpu_resolve(device, cmd->b, &b_off, &b_dt))) return st;
    b_off += cmd->b_byte_offset;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), bs = emunpu_dtype_size(b_dt),
         cs = emunpu_dtype_size(c_dt);

  uint32_t n = cmd->M;
  uint32_t period = cmd->bcast_period;
  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t start = 0; start < n; start += IREE_EMUNPU_VECTOR_LANES) {
    uint32_t len = n - start;
    if (len > IREE_EMUNPU_VECTOR_LANES) len = IREE_EMUNPU_VECTOR_LANES;
    emunpu_instr_t in = {0};
    in.op = EMUNPU_I_EW;
    in.a_dtype = a_dt;
    in.b_dtype = b_dt;
    in.c_dtype = c_dt;
    in.M = len;
    in.flags = kind;
    in.bcast_period = period;
    in.bcast_phase = period ? start % period : 0;
    in.a_space = EMUNPU_SPACE_GLOBAL;
    in.a_offset = a_off + (uint64_t)start * as;
    in.c_space = EMUNPU_SPACE_GLOBAL;
    in.c_offset = c_off + (uint64_t)start * cs;
    if (binary) {
      in.b_space = EMUNPU_SPACE_GLOBAL;
      // A broadcast operand is addressed from its base (the phase selects the
      // element); an equal-shape one advances with the slice.
      in.b_offset = period ? b_off : b_off + (uint64_t)start * bs;
    }
    if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
  }
  return emunpu_run_all_cores(device);
}

// Fused multiply-add / dtype conversion: VECTOR_LANES elements per instruction.
static emunpu_status_t emunpu_exec_vector2(emunpu_device_t* device,
                                           const emunpu_command_t* cmd,
                                           emunpu_iopcode_t op) {
  int has_b = (op == EMUNPU_I_FMA);
  uint64_t a_off, b_off = 0, c_off;
  emunpu_dtype_t a_dt, b_dt = EMUNPU_DT_FP32, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  if (has_b) {
    if ((st = emunpu_resolve(device, cmd->b, &b_off, &b_dt))) return st;
    b_off += cmd->b_byte_offset;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), bs = emunpu_dtype_size(b_dt),
         cs = emunpu_dtype_size(c_dt);

  uint32_t n = cmd->M;
  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t start = 0; start < n; start += IREE_EMUNPU_VECTOR_LANES) {
    uint32_t len = n - start;
    if (len > IREE_EMUNPU_VECTOR_LANES) len = IREE_EMUNPU_VECTOR_LANES;
    emunpu_instr_t in = {0};
    in.op = op;
    in.a_dtype = a_dt;
    in.b_dtype = b_dt;
    in.c_dtype = c_dt;
    in.M = len;
    in.a_space = in.b_space = in.c_space = EMUNPU_SPACE_GLOBAL;
    in.a_offset = a_off + (uint64_t)start * as;
    in.c_offset = c_off + (uint64_t)start * cs;
    if (has_b) in.b_offset = b_off + (uint64_t)start * bs;
    if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
  }
  return emunpu_run_all_cores(device);
}

// Matrix-vector: the matrix engine takes at most an MMA_DIM x MMA_DIM tile, so
// this blocks both M and K. The K-tiles of one row block accumulate into the
// same y elements, so they are pinned to a single core in order.
static emunpu_status_t emunpu_exec_matvec(emunpu_device_t* device,
                                          const emunpu_command_t* cmd) {
  uint64_t a_off, x_off, y_off;
  emunpu_dtype_t a_dt, x_dt, y_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->b, &x_off, &x_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &y_off, &y_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  x_off += cmd->b_byte_offset;
  y_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), xs = emunpu_dtype_size(x_dt),
         ys = emunpu_dtype_size(y_dt);
  uint32_t M = cmd->M, K = cmd->K;
  const uint32_t tile = IREE_EMUNPU_MMA_DIM;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  uint32_t block = 0;
  for (uint32_t m0 = 0; m0 < M; m0 += tile, ++block) {
    uint32_t rows = (M - m0 < tile) ? (M - m0) : tile;
    for (uint32_t k0 = 0; k0 < K; k0 += tile) {
      uint32_t kt = (K - k0 < tile) ? (K - k0) : tile;
      emunpu_instr_t in = {0};
      in.op = EMUNPU_I_MATVEC;
      in.a_dtype = a_dt;
      in.b_dtype = x_dt;
      in.c_dtype = y_dt;
      in.M = rows;
      in.K = kt;
      in.flags = k0 ? EMUNPU_MMA_ACCUMULATE : 0;
      in.a_space = in.b_space = in.c_space = EMUNPU_SPACE_GLOBAL;
      in.a_offset = a_off + ((uint64_t)m0 * K + k0) * as;
      in.a_row_stride = (uint32_t)((uint64_t)K * as);
      in.b_offset = x_off + (uint64_t)k0 * xs;
      in.c_offset = y_off + (uint64_t)m0 * ys;
      if ((st = emunpu_issue_to(&it, block, &in)) != EMUNPU_OK) return st;
    }
  }
  return emunpu_run_all_cores(device);
}

//===----------------------------------------------------------------------===//
// ML operation execution
//===----------------------------------------------------------------------===//

// Activations run on the narrower special-function unit: SFU_LANES per issue.
static emunpu_status_t emunpu_exec_act(emunpu_device_t* device,
                                       const emunpu_command_t* cmd) {
  uint64_t a_off, c_off;
  emunpu_dtype_t a_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), cs = emunpu_dtype_size(c_dt);
  uint32_t n = cmd->M;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t start = 0; start < n; start += IREE_EMUNPU_SFU_LANES) {
    uint32_t len = n - start;
    if (len > IREE_EMUNPU_SFU_LANES) len = IREE_EMUNPU_SFU_LANES;
    emunpu_instr_t in = {0};
    in.op = EMUNPU_I_ACT;
    in.a_dtype = a_dt;
    in.c_dtype = c_dt;
    in.M = len;
    in.flags = cmd->modifier;
    in.param0 = cmd->param0;
    in.param1 = cmd->param1;
    in.a_space = in.c_space = EMUNPU_SPACE_GLOBAL;
    in.a_offset = a_off + (uint64_t)start * as;
    in.c_offset = c_off + (uint64_t)start * cs;
    if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
  }
  return emunpu_run_all_cores(device);
}

// Row-wise ops (REDUCE / SOFTMAX / NORM). The row unit streams one row through
// a single accumulator per issue, so this emits one instruction per row and
// deals the rows across the cores. A row wider than the unit's row buffer
// cannot be expressed at all -- the caller has to split the tensor.
static emunpu_status_t emunpu_exec_rowwise(emunpu_device_t* device,
                                           const emunpu_command_t* cmd,
                                           emunpu_iopcode_t op) {
  uint64_t a_off, c_off;
  emunpu_dtype_t a_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), cs = emunpu_dtype_size(c_dt);
  uint32_t rows = cmd->M, width = cmd->N;
  if (width > IREE_EMUNPU_MAX_ROW_WIDTH) return EMUNPU_ERROR_OUT_OF_RANGE;
  // A reduction collapses each row to one element; softmax/norm keep the width.
  uint32_t c_per_row = (op == EMUNPU_I_REDUCE) ? 1 : width;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t r = 0; r < rows; ++r) {
    emunpu_instr_t in = {0};
    in.op = op;
    in.a_dtype = a_dt;
    in.c_dtype = c_dt;
    in.M = 1;
    in.N = width;
    in.flags = cmd->modifier;
    in.param0 = cmd->param0;
    in.a_space = in.c_space = EMUNPU_SPACE_GLOBAL;
    in.a_offset = a_off + (uint64_t)r * width * as;
    in.c_offset = c_off + (uint64_t)r * c_per_row * cs;
    if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
  }
  return emunpu_run_all_cores(device);
}

// Transpose runs through the matrix engine's tile shuffler, so it is blocked
// into MMA_DIM x MMA_DIM tiles; each instruction reads one source tile and
// writes the transposed tile via the operands' row strides.
static emunpu_status_t emunpu_exec_transpose(emunpu_device_t* device,
                                             const emunpu_command_t* cmd) {
  uint64_t a_off, c_off;
  emunpu_dtype_t a_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), cs = emunpu_dtype_size(c_dt);
  uint32_t rows = cmd->M, cols = cmd->N;
  const uint32_t tile = IREE_EMUNPU_MMA_DIM;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t r0 = 0; r0 < rows; r0 += tile) {
    uint32_t m = (rows - r0 < tile) ? (rows - r0) : tile;
    for (uint32_t c0 = 0; c0 < cols; c0 += tile) {
      uint32_t n = (cols - c0 < tile) ? (cols - c0) : tile;
      emunpu_instr_t in = {0};
      in.op = EMUNPU_I_TRANSPOSE;
      in.a_dtype = a_dt;
      in.c_dtype = c_dt;
      in.M = m;
      in.N = n;
      in.a_space = in.c_space = EMUNPU_SPACE_GLOBAL;
      in.a_offset = a_off + ((uint64_t)r0 * cols + c0) * as;
      in.a_row_stride = (uint32_t)((uint64_t)cols * as);
      in.c_offset = c_off + ((uint64_t)c0 * rows + r0) * cs;
      in.c_row_stride = (uint32_t)((uint64_t)rows * cs);
      if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
    }
  }
  return emunpu_run_all_cores(device);
}

// im2col emits at most one MMA operand tile of patch rows per instruction, for
// each image of the batch.
static emunpu_status_t emunpu_exec_im2col(emunpu_device_t* device,
                                          const emunpu_command_t* cmd) {
  uint64_t a_off, c_off;
  emunpu_dtype_t a_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), cs = emunpu_dtype_size(c_dt);
  const emunpu_spatial_t* sp = &cmd->spatial;
  uint32_t batch = cmd->K;  // K carries the batch size for this command
  uint32_t patches = sp->out_h * sp->out_w;
  uint32_t width = cmd->N;
  if (width > IREE_EMUNPU_MAX_ROW_WIDTH) return EMUNPU_ERROR_OUT_OF_RANGE;
  uint64_t image_bytes = (uint64_t)sp->channels * sp->in_h * sp->in_w * as;
  uint64_t image_patch_bytes = (uint64_t)patches * width * cs;
  const uint32_t rows_per = IREE_EMUNPU_MMA_DIM;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t bi = 0; bi < batch; ++bi) {
    for (uint32_t start = 0; start < patches; start += rows_per) {
      uint32_t r = (patches - start < rows_per) ? (patches - start) : rows_per;
      emunpu_instr_t in = {0};
      in.op = EMUNPU_I_IM2COL;
      in.a_dtype = a_dt;
      in.c_dtype = c_dt;
      in.M = r;
      in.N = width;
      in.spatial = *sp;
      in.spatial.first_out_row = start;
      in.a_space = in.c_space = EMUNPU_SPACE_GLOBAL;
      in.a_offset = a_off + bi * image_bytes;
      in.c_offset =
          c_off + bi * image_patch_bytes + (uint64_t)start * width * cs;
      if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
    }
  }
  return emunpu_run_all_cores(device);
}

// Pooling: one instruction per row band of one CHW plane, sized so the rows it
// writes fit the vector unit.
static emunpu_status_t emunpu_exec_pool2d(emunpu_device_t* device,
                                          const emunpu_command_t* cmd) {
  uint64_t a_off, c_off;
  emunpu_dtype_t a_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t as = emunpu_dtype_size(a_dt), cs = emunpu_dtype_size(c_dt);
  const emunpu_spatial_t* sp = &cmd->spatial;
  uint32_t planes = cmd->M;
  if (sp->out_w > IREE_EMUNPU_VECTOR_LANES) return EMUNPU_ERROR_OUT_OF_RANGE;
  uint32_t rows_per = IREE_EMUNPU_VECTOR_LANES / sp->out_w;
  if (rows_per == 0) rows_per = 1;
  uint64_t in_plane = (uint64_t)sp->in_h * sp->in_w * as;
  uint64_t out_plane = (uint64_t)sp->out_h * sp->out_w * cs;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t p = 0; p < planes; ++p) {
    for (uint32_t r0 = 0; r0 < sp->out_h; r0 += rows_per) {
      uint32_t m = (sp->out_h - r0 < rows_per) ? (sp->out_h - r0) : rows_per;
      emunpu_instr_t in = {0};
      in.op = EMUNPU_I_POOL2D;
      in.a_dtype = a_dt;
      in.c_dtype = c_dt;
      in.M = m;
      in.flags = cmd->modifier;
      in.spatial = *sp;
      in.spatial.first_out_row = r0;
      in.a_space = in.c_space = EMUNPU_SPACE_GLOBAL;
      in.a_offset = a_off + p * in_plane;
      in.c_offset = c_off + p * out_plane + (uint64_t)r0 * sp->out_w * cs;
      if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
    }
  }
  return emunpu_run_all_cores(device);
}

// Gather: up to MMA_DIM rows per instruction.
static emunpu_status_t emunpu_exec_gather(emunpu_device_t* device,
                                          const emunpu_command_t* cmd) {
  uint64_t a_off, b_off, c_off;
  emunpu_dtype_t a_dt, b_dt, c_dt;
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, cmd->a, &a_off, &a_dt)) ||
      (st = emunpu_resolve(device, cmd->b, &b_off, &b_dt)) ||
      (st = emunpu_resolve(device, cmd->c, &c_off, &c_dt))) {
    return st;
  }
  a_off += cmd->a_byte_offset;
  b_off += cmd->b_byte_offset;
  c_off += cmd->c_byte_offset;
  size_t bs = emunpu_dtype_size(b_dt), cs = emunpu_dtype_size(c_dt);
  uint32_t count = cmd->M, width = cmd->N;
  if (width > IREE_EMUNPU_MAX_ROW_WIDTH) return EMUNPU_ERROR_OUT_OF_RANGE;
  const uint32_t rows_per = IREE_EMUNPU_MMA_DIM;

  emunpu_issue_t it;
  emunpu_issue_begin(&it, device, cmd);
  for (uint32_t start = 0; start < count; start += rows_per) {
    uint32_t len = (count - start < rows_per) ? (count - start) : rows_per;
    emunpu_instr_t in = {0};
    in.op = EMUNPU_I_GATHER;
    in.a_dtype = a_dt;
    in.b_dtype = b_dt;
    in.c_dtype = c_dt;
    in.M = len;
    in.N = width;
    in.K = cmd->K;  // table row count (bounds check)
    in.a_space = in.b_space = in.c_space = EMUNPU_SPACE_GLOBAL;
    in.a_offset = a_off;
    in.b_offset = b_off + (uint64_t)start * bs;
    in.c_offset = c_off + (uint64_t)start * width * cs;
    if ((st = emunpu_issue(&it, &in)) != EMUNPU_OK) return st;
  }
  return emunpu_run_all_cores(device);
}

static emunpu_status_t emunpu_exec_command(emunpu_device_t* device,
                                           const emunpu_command_t* cmd) {
  switch (cmd->kind) {
    case EMUNPU_CMD_MATMUL:
      return emunpu_exec_matmul(device, cmd);
    case EMUNPU_CMD_MATVEC:
      return emunpu_exec_matvec(device, cmd);
    case EMUNPU_CMD_EW:
      return emunpu_exec_ew(device, cmd);
    case EMUNPU_CMD_REDUCE:
      return emunpu_exec_rowwise(device, cmd, EMUNPU_I_REDUCE);
    case EMUNPU_CMD_FMA:
      return emunpu_exec_vector2(device, cmd, EMUNPU_I_FMA);
    case EMUNPU_CMD_CONVERT:
      return emunpu_exec_vector2(device, cmd, EMUNPU_I_CONVERT);
    case EMUNPU_CMD_ACT:
      return emunpu_exec_act(device, cmd);
    case EMUNPU_CMD_SOFTMAX:
      return emunpu_exec_rowwise(device, cmd, EMUNPU_I_SOFTMAX);
    case EMUNPU_CMD_NORM:
      return emunpu_exec_rowwise(device, cmd, EMUNPU_I_NORM);
    case EMUNPU_CMD_TRANSPOSE:
      return emunpu_exec_transpose(device, cmd);
    case EMUNPU_CMD_IM2COL:
      return emunpu_exec_im2col(device, cmd);
    case EMUNPU_CMD_POOL2D:
      return emunpu_exec_pool2d(device, cmd);
    case EMUNPU_CMD_GATHER:
      return emunpu_exec_gather(device, cmd);
    case EMUNPU_CMD_FREE:
      return emunpu_global_mem_free(&device->gm, cmd->c);
    default:
      return EMUNPU_ERROR_UNIMPLEMENTED;
  }
}

//===----------------------------------------------------------------------===//
// Streams + events
//===----------------------------------------------------------------------===//

static emunpu_status_t emunpu_device_register_stream(emunpu_device_t* device,
                                                     emunpu_stream_t* stream) {
  if (device->stream_count == device->stream_cap) {
    uint32_t cap = device->stream_cap ? device->stream_cap * 2 : 4;
    emunpu_stream_t** grown = (emunpu_stream_t**)realloc(
        device->streams, cap * sizeof(emunpu_stream_t*));
    if (!grown) return EMUNPU_ERROR_OUT_OF_MEMORY;
    device->streams = grown;
    device->stream_cap = cap;
  }
  device->streams[device->stream_count++] = stream;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_stream_create(emunpu_device_t* device,
                                     emunpu_stream_t** out_stream) {
  if (!device || !out_stream) return EMUNPU_ERROR_INVALID_ARGUMENT;
  *out_stream = NULL;
  emunpu_stream_t* stream = (emunpu_stream_t*)calloc(1, sizeof(*stream));
  if (!stream) return EMUNPU_ERROR_OUT_OF_MEMORY;
  stream->device = device;
  stream->cap = 8;
  stream->cmds =
      (emunpu_command_t*)malloc(stream->cap * sizeof(emunpu_command_t));
  if (!stream->cmds) {
    free(stream);
    return EMUNPU_ERROR_OUT_OF_MEMORY;
  }
  emunpu_status_t st = emunpu_device_register_stream(device, stream);
  if (st != EMUNPU_OK) {
    free(stream->cmds);
    free(stream);
    return st;
  }
  *out_stream = stream;
  return EMUNPU_OK;
}

void emunpu_stream_destroy(emunpu_stream_t* stream) {
  if (!stream) return;
  emunpu_device_t* device = stream->device;
  for (uint32_t i = 0; i < device->stream_count; ++i) {
    if (device->streams[i] == stream) {
      device->streams[i] = device->streams[--device->stream_count];
      break;
    }
  }
  free(stream->cmds);
  free(stream);
}

static emunpu_status_t emunpu_stream_append(emunpu_stream_t* stream,
                                            const emunpu_command_t* cmd) {
  if (stream->len == stream->cap) {
    uint32_t cap = stream->cap * 2;
    emunpu_command_t* grown = (emunpu_command_t*)realloc(
        stream->cmds, cap * sizeof(emunpu_command_t));
    if (!grown) return EMUNPU_ERROR_OUT_OF_MEMORY;
    stream->cmds = grown;
    stream->cap = cap;
  }
  stream->cmds[stream->len++] = *cmd;
  return EMUNPU_OK;
}

// Submits |cmd| to |stream| (deferred) or executes it now (stream == NULL),
// stamping it with the execution scope in force right now.
static emunpu_status_t emunpu_submit(emunpu_device_t* device,
                                     emunpu_stream_t* stream,
                                     const emunpu_command_t* cmd) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t scoped = *cmd;
  scoped.scope_first = device->scope_first;
  scoped.scope_count = device->scope_count ? device->scope_count : 1;
  if (!stream) return emunpu_exec_command(device, &scoped);
  return emunpu_stream_append(stream, &scoped);
}

emunpu_status_t emunpu_stream_synchronize(emunpu_stream_t* stream) {
  if (!stream) return EMUNPU_OK;
  while (stream->executed < stream->len) {
    emunpu_status_t st =
        emunpu_exec_command(stream->device, &stream->cmds[stream->executed]);
    if (st != EMUNPU_OK) return st;
    stream->executed++;
  }
  // All drained; reset so the stream can be reused.
  stream->len = 0;
  stream->executed = 0;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_event_create(emunpu_device_t* device,
                                    emunpu_event_t** out_event) {
  if (!device || !out_event) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_event_t* event = (emunpu_event_t*)calloc(1, sizeof(*event));
  if (!event) return EMUNPU_ERROR_OUT_OF_MEMORY;
  event->device = device;
  *out_event = event;
  return EMUNPU_OK;
}

void emunpu_event_destroy(emunpu_event_t* event) { free(event); }

emunpu_status_t emunpu_event_record(emunpu_event_t* event,
                                    emunpu_stream_t* stream) {
  if (!event) return EMUNPU_ERROR_INVALID_ARGUMENT;
  event->stream = stream;
  event->recorded_pos = stream ? stream->len : 0;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_event_synchronize(emunpu_event_t* event) {
  if (!event) return EMUNPU_ERROR_INVALID_ARGUMENT;
  // Functional model: draining the recording stream guarantees the work up to
  // (and past) the recorded position has executed.
  return event->stream ? emunpu_stream_synchronize(event->stream) : EMUNPU_OK;
}

emunpu_status_t emunpu_stream_wait_event(emunpu_stream_t* stream,
                                         emunpu_event_t* event) {
  if (!event) return EMUNPU_ERROR_INVALID_ARGUMENT;
  // Ensure the event's producer work has executed before |stream| continues.
  // (Single-threaded model: draining now preserves the cross-stream ordering.)
  (void)stream;
  return emunpu_event_synchronize(event);
}

emunpu_status_t emunpu_device_synchronize(emunpu_device_t* device) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < device->stream_count; ++i) {
    emunpu_status_t st = emunpu_stream_synchronize(device->streams[i]);
    if (st != EMUNPU_OK) return st;
  }
  return EMUNPU_OK;
}

emunpu_status_t emunpu_device_barrier(emunpu_device_t* device) {
  if (device && device->trace) {
    fprintf(stderr, "[npu] barrier: syncing %u streams across %u cores\n",
            device->stream_count, device ? device->num_cores : 0);
  }
  if (device) emunpu_viz_barrier(device->stream_count);
  return emunpu_device_synchronize(device);
}

//===----------------------------------------------------------------------===//
// Compute API
//===----------------------------------------------------------------------===//

emunpu_status_t emunpu_matmul(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              emunpu_tensor_id_t b, uint32_t M, uint32_t N,
                              uint32_t K, int transpose_b) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_MATMUL;
  cmd.a = a;
  cmd.b = b;
  cmd.c = c;
  cmd.M = M;
  cmd.N = N;
  cmd.K = K;
  cmd.modifier = (uint32_t)(transpose_b ? 1 : 0);
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_matmul_scheduled(
    emunpu_device_t* device, emunpu_tensor_id_t c, emunpu_tensor_id_t a,
    emunpu_tensor_id_t b, uint32_t M, uint32_t N, uint32_t K, int transpose_b,
    const emunpu_gemm_tile_t* tiles, uint32_t tile_count) {
  if (!device) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_gemm_desc_t g = {0};
  emunpu_status_t st;
  if ((st = emunpu_resolve(device, a, &g.a_offset, &g.a_dtype)) ||
      (st = emunpu_resolve(device, b, &g.b_offset, &g.b_dtype)) ||
      (st = emunpu_resolve(device, c, &g.c_offset, &g.c_dtype))) {
    return st;
  }
  g.M = M;
  g.N = N;
  g.K = K;
  g.transpose_b = transpose_b;

  // Replay the compiler's schedule rather than tiling on-device. The schedule's
  // core ids are folded onto the execution scope, so a narrowed scope runs a
  // schedule built for a wider device.
  st = emunpu_run_gemm_schedule(device->cores + device->scope_first,
                                device->scope_count, &g, tiles, tile_count);
  if (st != EMUNPU_OK) return st;
  emunpu_viz_gemm(g.M, g.N, g.K, g.transpose_b, tile_count,
                  device->scope_count);
  if (device->trace) {
    fprintf(stderr,
            "[npu] GEMM %ux%ux%u %s-> %u compiler-scheduled tiles across %u "
            "cores\n",
            g.M, g.N, g.K, g.transpose_b ? "transpose_b " : "", tile_count,
            device->scope_count);
  }
  return emunpu_run_all_cores(device);
}

emunpu_status_t emunpu_gemm(emunpu_device_t* device, emunpu_stream_t* stream,
                            emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                            emunpu_tensor_id_t b, emunpu_tensor_id_t bias,
                            uint32_t M, uint32_t N, uint32_t K, int transpose_b,
                            int fuse_relu) {
  emunpu_status_t st =
      emunpu_matmul(device, stream, c, a, b, M, N, K, transpose_b);
  if (st != EMUNPU_OK) return st;
  if (bias) {
    // C = C + bias, with bias broadcast across each row (period N).
    emunpu_command_t add = {0};
    add.kind = EMUNPU_CMD_EW;
    add.a = c;
    add.b = bias;
    add.c = c;
    add.M = M * N;
    add.modifier = EMUNPU_EW_ADD;
    add.bcast_period = N;
    if ((st = emunpu_submit(device, stream, &add)) != EMUNPU_OK) return st;
  }
  if (fuse_relu) {
    emunpu_command_t relu = {0};
    relu.kind = EMUNPU_CMD_EW;
    relu.a = c;
    relu.c = c;
    relu.M = M * N;
    relu.modifier = EMUNPU_EW_RELU;
    if ((st = emunpu_submit(device, stream, &relu)) != EMUNPU_OK) return st;
  }
  return EMUNPU_OK;
}

emunpu_status_t emunpu_matvec(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t y, emunpu_tensor_id_t a,
                              emunpu_tensor_id_t x, uint32_t M, uint32_t K) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_MATVEC;
  cmd.a = a;
  cmd.b = x;
  cmd.c = y;
  cmd.M = M;
  cmd.K = K;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_vector_op(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, emunpu_tensor_id_t b,
                                 uint32_t count, emunpu_ew_kind_t kind,
                                 uint32_t bcast_period) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_EW;
  cmd.a = a;
  cmd.b = b;
  cmd.c = c;
  cmd.M = count;
  cmd.modifier = kind;
  cmd.bcast_period = bcast_period;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_reduce(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              uint32_t rows, uint32_t width,
                              emunpu_reduce_kind_t kind) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_REDUCE;
  cmd.a = a;
  cmd.c = c;
  cmd.M = rows;
  cmd.N = width;
  cmd.modifier = kind;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_fma(emunpu_device_t* device, emunpu_stream_t* stream,
                           emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                           emunpu_tensor_id_t b, uint32_t count) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_FMA;
  cmd.a = a;
  cmd.b = b;
  cmd.c = c;
  cmd.M = count;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_convert(emunpu_device_t* device, emunpu_stream_t* stream,
                               emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                               uint32_t count) {
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_CONVERT;
  cmd.a = a;
  cmd.c = c;
  cmd.M = count;
  return emunpu_submit(device, stream, &cmd);
}

//===----------------------------------------------------------------------===//
// Machine-learning operations
//===----------------------------------------------------------------------===//

void emunpu_window_output(const emunpu_window_t* window, uint32_t* out_h,
                          uint32_t* out_w) {
  uint32_t h = 0, w = 0;
  if (window) {
    h = emunpu_spatial_out_extent(window->in_h, window->kernel_h,
                                  window->stride_h, window->pad_h);
    w = emunpu_spatial_out_extent(window->in_w, window->kernel_w,
                                  window->stride_w, window->pad_w);
  }
  if (out_h) *out_h = h;
  if (out_w) *out_w = w;
}

// Fills the micro-ISA spatial descriptor from a public window description,
// deriving (and validating) the output extents. Returns INVALID_ARGUMENT for a
// window that does not fit the input.
static emunpu_status_t emunpu_window_to_spatial(const emunpu_window_t* window,
                                                emunpu_spatial_t* out_spatial) {
  if (!window || !window->batch || !window->channels) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  uint32_t out_h = 0, out_w = 0;
  emunpu_window_output(window, &out_h, &out_w);
  if (!out_h || !out_w) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_spatial_t sp = {0};
  sp.in_h = window->in_h;
  sp.in_w = window->in_w;
  sp.channels = window->channels;
  sp.kernel_h = window->kernel_h;
  sp.kernel_w = window->kernel_w;
  sp.stride_h = window->stride_h;
  sp.stride_w = window->stride_w;
  sp.pad_h = window->pad_h;
  sp.pad_w = window->pad_w;
  sp.out_h = out_h;
  sp.out_w = out_w;
  *out_spatial = sp;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_activation(emunpu_device_t* device,
                                  emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                  emunpu_tensor_id_t a, uint32_t count,
                                  emunpu_act_kind_t kind, double alpha,
                                  double beta) {
  if (kind >= EMUNPU_ACT_COUNT) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_ACT;
  cmd.a = a;
  cmd.c = c;
  cmd.M = count;
  cmd.modifier = (uint32_t)kind;
  cmd.param0 = alpha;
  cmd.param1 = beta;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_linear(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              emunpu_tensor_id_t b, emunpu_tensor_id_t bias,
                              uint32_t M, uint32_t N, uint32_t K,
                              int transpose_b, const emunpu_act_t* act) {
  if (act && act->kind >= EMUNPU_ACT_COUNT) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  // The matmul + broadcast bias add is exactly emunpu_gemm's body; only the
  // epilogue generalizes from ReLU to any activation kind.
  emunpu_status_t st =
      emunpu_gemm(device, stream, c, a, b, bias, M, N, K, transpose_b,
                  /*fuse_relu=*/0);
  if (st != EMUNPU_OK || !act) return st;
  return emunpu_activation(device, stream, c, c, M * N, act->kind, act->alpha,
                           act->beta);
}

emunpu_status_t emunpu_softmax(emunpu_device_t* device, emunpu_stream_t* stream,
                               emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                               uint32_t rows, uint32_t width, int log_softmax) {
  if (!rows || !width) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_SOFTMAX;
  cmd.a = a;
  cmd.c = c;
  cmd.M = rows;
  cmd.N = width;
  cmd.modifier = log_softmax ? EMUNPU_SOFTMAX_LOG : 0;
  return emunpu_submit(device, stream, &cmd);
}

// Shared body of layernorm / rmsnorm: normalize, then apply the optional affine
// terms as period-|width| broadcast elementwise ops.
static emunpu_status_t emunpu_normalize(
    emunpu_device_t* device, emunpu_stream_t* stream, emunpu_tensor_id_t c,
    emunpu_tensor_id_t a, emunpu_tensor_id_t gamma, emunpu_tensor_id_t beta,
    uint32_t rows, uint32_t width, double eps, emunpu_norm_kind_t kind) {
  if (!rows || !width) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_NORM;
  cmd.a = a;
  cmd.c = c;
  cmd.M = rows;
  cmd.N = width;
  cmd.modifier = (uint32_t)kind;
  cmd.param0 = eps;
  emunpu_status_t st = emunpu_submit(device, stream, &cmd);
  if (st != EMUNPU_OK) return st;
  if (gamma) {
    st = emunpu_vector_op(device, stream, c, c, gamma, rows * width,
                          EMUNPU_EW_MUL, width);
    if (st != EMUNPU_OK) return st;
  }
  if (beta) {
    st = emunpu_vector_op(device, stream, c, c, beta, rows * width,
                          EMUNPU_EW_ADD, width);
    if (st != EMUNPU_OK) return st;
  }
  return EMUNPU_OK;
}

emunpu_status_t emunpu_layernorm(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, emunpu_tensor_id_t gamma,
                                 emunpu_tensor_id_t beta, uint32_t rows,
                                 uint32_t width, double eps) {
  return emunpu_normalize(device, stream, c, a, gamma, beta, rows, width, eps,
                          EMUNPU_NORM_LAYER);
}

emunpu_status_t emunpu_rmsnorm(emunpu_device_t* device, emunpu_stream_t* stream,
                               emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                               emunpu_tensor_id_t gamma, uint32_t rows,
                               uint32_t width, double eps) {
  return emunpu_normalize(device, stream, c, a, gamma, /*beta=*/0, rows, width,
                          eps, EMUNPU_NORM_RMS);
}

emunpu_status_t emunpu_transpose(emunpu_device_t* device,
                                 emunpu_stream_t* stream, emunpu_tensor_id_t c,
                                 emunpu_tensor_id_t a, uint32_t rows,
                                 uint32_t cols) {
  if (!rows || !cols) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_TRANSPOSE;
  cmd.a = a;
  cmd.c = c;
  cmd.M = rows;
  cmd.N = cols;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_pool2d(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              const emunpu_window_t* window,
                              emunpu_pool_kind_t kind) {
  if (kind >= EMUNPU_POOL_COUNT) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_spatial_t sp;
  emunpu_status_t st = emunpu_window_to_spatial(window, &sp);
  if (st != EMUNPU_OK) return st;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_POOL2D;
  cmd.a = a;
  cmd.c = c;
  cmd.M = window->batch * window->channels;  // independent CHW planes
  cmd.modifier = (uint32_t)kind;
  cmd.spatial = sp;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_im2col(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t a,
                              const emunpu_window_t* window) {
  emunpu_spatial_t sp;
  emunpu_status_t st = emunpu_window_to_spatial(window, &sp);
  if (st != EMUNPU_OK) return st;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_IM2COL;
  cmd.a = a;
  cmd.c = c;
  cmd.M = window->batch * sp.out_h * sp.out_w;  // total patch rows
  cmd.N = window->channels * window->kernel_h * window->kernel_w;
  cmd.K = window->batch;
  cmd.spatial = sp;
  return emunpu_submit(device, stream, &cmd);
}

emunpu_status_t emunpu_conv2d(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t out, emunpu_tensor_id_t in,
                              emunpu_tensor_id_t weight,
                              emunpu_tensor_id_t bias,
                              const emunpu_window_t* window,
                              uint32_t out_channels, const emunpu_act_t* act) {
  if (!device || !out_channels) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (act && act->kind >= EMUNPU_ACT_COUNT) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;
  }
  emunpu_spatial_t sp;
  emunpu_status_t st = emunpu_window_to_spatial(window, &sp);
  if (st != EMUNPU_OK) return st;

  const emunpu_tensor_desc_t* in_desc =
      emunpu_global_mem_lookup(&device->gm, in);
  const emunpu_tensor_desc_t* out_desc =
      emunpu_global_mem_lookup(&device->gm, out);
  if (!in_desc || !out_desc) return EMUNPU_ERROR_NOT_FOUND;

  uint32_t patches = sp.out_h * sp.out_w;   // per image
  uint32_t rows = window->batch * patches;  // GEMM M
  uint32_t width = window->channels * window->kernel_h * window->kernel_w;

  // Scratch: the patch matrix, and the GEMM result in [rows, out_channels]
  // layout (i.e. channel-last) which the final transpose turns back into NCHW.
  emunpu_tensor_id_t patch_buf = 0, gemm_buf = 0;
  uint32_t patch_shape[2] = {rows, width};
  uint32_t gemm_shape[2] = {rows, out_channels};
  st = emunpu_device_alloc(device, in_desc->dtype, 2, patch_shape,
                           "conv.patches", &patch_buf);
  if (st != EMUNPU_OK) return st;
  st = emunpu_device_alloc(device, out_desc->dtype, 2, gemm_shape, "conv.gemm",
                           &gemm_buf);
  if (st != EMUNPU_OK) {
    emunpu_device_free(device, patch_buf);
    return st;
  }

  // 1. im2col: NCHW image -> patch rows.
  if ((st = emunpu_im2col(device, stream, patch_buf, in, window)) !=
      EMUNPU_OK) {
    goto cleanup;
  }
  // 2. the convolution proper, as a GEMM on the tensor cores. The weight is
  //    [out_channels, width], so this is a transpose_b matmul.
  if ((st = emunpu_matmul(device, stream, gemm_buf, patch_buf, weight, rows,
                          out_channels, width, /*transpose_b=*/1)) !=
      EMUNPU_OK) {
    goto cleanup;
  }
  // 3. + bias (broadcast per output channel) and 4. the activation epilogue.
  if (bias) {
    st = emunpu_vector_op(device, stream, gemm_buf, gemm_buf, bias,
                          rows * out_channels, EMUNPU_EW_ADD, out_channels);
    if (st != EMUNPU_OK) goto cleanup;
  }
  if (act) {
    st = emunpu_activation(device, stream, gemm_buf, gemm_buf,
                           rows * out_channels, act->kind, act->alpha,
                           act->beta);
    if (st != EMUNPU_OK) goto cleanup;
  }
  // 5. [patches, out_channels] -> [out_channels, patches] per image, which is
  //    the NCHW output layout.
  for (uint32_t bi = 0; bi < window->batch; ++bi) {
    emunpu_command_t t = {0};
    t.kind = EMUNPU_CMD_TRANSPOSE;
    t.a = gemm_buf;
    t.c = out;
    t.M = patches;
    t.N = out_channels;
    t.a_byte_offset = (uint64_t)bi * patches * out_channels *
                      emunpu_dtype_size(out_desc->dtype);
    t.c_byte_offset = (uint64_t)bi * out_channels * patches *
                      emunpu_dtype_size(out_desc->dtype);
    if ((st = emunpu_submit(device, stream, &t)) != EMUNPU_OK) goto cleanup;
  }

cleanup:
  // Release the scratch after the work that uses it: right now on the default
  // stream (everything above already executed), or as a queued FREE command
  // ordered behind the deferred commands that read it.
  for (int i = 0; i < 2; ++i) {
    emunpu_tensor_id_t id = i == 0 ? gemm_buf : patch_buf;
    emunpu_status_t freed = EMUNPU_ERROR_FAILED_PRECONDITION;
    if (stream) {
      emunpu_command_t f = {0};
      f.kind = EMUNPU_CMD_FREE;
      f.c = id;
      freed = emunpu_stream_append(stream, &f);
    }
    if (freed != EMUNPU_OK) emunpu_device_free(device, id);
  }
  return st;
}

emunpu_status_t emunpu_gather(emunpu_device_t* device, emunpu_stream_t* stream,
                              emunpu_tensor_id_t c, emunpu_tensor_id_t table,
                              emunpu_tensor_id_t indices, uint32_t table_rows,
                              uint32_t count, uint32_t width) {
  if (!table_rows || !width) return EMUNPU_ERROR_INVALID_ARGUMENT;
  emunpu_command_t cmd = {0};
  cmd.kind = EMUNPU_CMD_GATHER;
  cmd.a = table;
  cmd.b = indices;
  cmd.c = c;
  cmd.M = count;
  cmd.N = width;
  cmd.K = table_rows;
  return emunpu_submit(device, stream, &cmd);
}
