// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// A single emulated NPU compute core.
//
// Each core owns: a private local scratchpad ("local memory"), a private
// command queue of micro-instructions, and a tensor-core-style compute engine
// that executes them. Operands live either in global memory (the shared arena,
// reached via DMA) or in the core's local scratchpad. A core is otherwise
// independent of the others -- the device fans work out across the
// IREE_EMUNPU_NUM_CORES cores by filling their queues.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_CORE_H_
#define EMUNPU_CORE_H_

#include "emunpu/arch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct emunpu_core_t {
  uint32_t core_id;

  // Local scratchpad memory. The engine DMAs tiles here from global memory,
  // computes against them, and DMAs results back out.
  uint8_t* local_mem;
  uint64_t local_capacity;
  uint64_t local_watermark;  // simple per-dispatch bump allocator

  // Private command queue (grown on demand).
  emunpu_instr_t* queue;
  uint32_t queue_len;
  uint32_t queue_cap;

  // Execution statistics (useful for the trace / scheduling diagnostics).
  uint64_t instr_count;
  uint64_t mma_count;
  uint64_t dma_bytes;
} emunpu_core_t;

EMUNPU_API emunpu_status_t emunpu_core_initialize(emunpu_core_t* core, uint32_t core_id);
EMUNPU_API void emunpu_core_deinitialize(emunpu_core_t* core);

// Bump-allocates |bytes| of local scratchpad, returning its local byte offset.
EMUNPU_API emunpu_status_t emunpu_core_local_alloc(emunpu_core_t* core, uint64_t bytes,
                                        uint64_t* out_offset);

// Resets the local scratchpad bump allocator (frees all local tiles at once).
EMUNPU_API void emunpu_core_local_reset(emunpu_core_t* core);

// Appends an instruction to the core's queue.
EMUNPU_API emunpu_status_t emunpu_core_enqueue(emunpu_core_t* core,
                                    const emunpu_instr_t* instr);

// Clears the command queue (does not touch local memory).
EMUNPU_API void emunpu_core_queue_reset(emunpu_core_t* core);

// Executes every queued instruction in order against |global_arena|, then
// clears the queue. |trace| logs each instruction with the core id.
EMUNPU_API emunpu_status_t emunpu_core_run(emunpu_core_t* core, uint8_t* global_arena,
                                uint64_t global_capacity, int trace);

// Executes a single instruction (also used directly by tests). |trace| logs it.
EMUNPU_API emunpu_status_t emunpu_core_execute(emunpu_core_t* core,
                                    const emunpu_instr_t* instr,
                                    uint8_t* global_arena,
                                    uint64_t global_capacity, int trace);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_CORE_H_
