// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Centralized global memory subsystem.
//
// All tensors and buffers visible to the cores live in a single contiguous
// byte arena managed here. Each live allocation is tracked by a descriptor
// (id, dtype, shape, byte offset, size, name) in a descriptor table. The table
// is the device's authoritative *memory layout*, and can be serialized to /
// parsed from a FlatBuffer (see emunpu_mem.fbs + emunpu_global_mem_serialize)
// so the layout is inspectable and portable.
//
// Allocation is a coalescing free-list over the arena: freed regions are
// reused, otherwise the watermark bumps. Cores never allocate from here at
// runtime; they DMA in/out of regions the device handed them.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_GLOBAL_MEM_H_
#define EMUNPU_GLOBAL_MEM_H_

#include "emunpu/arch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque handle for an allocation in global memory. 0 is the null/invalid id.
typedef uint32_t emunpu_tensor_id_t;

// Descriptor for one live allocation. Shape is informational metadata; the
// allocator only cares about |offset| and |size|.
typedef struct emunpu_tensor_desc_t {
  emunpu_tensor_id_t id;
  emunpu_dtype_t dtype;
  uint32_t rank;
  uint32_t shape[IREE_EMUNPU_MAX_RANK];
  uint64_t offset;  // byte offset into the arena
  uint64_t size;    // byte size of the allocation
  char name[32];    // optional debug label (NUL-terminated)
} emunpu_tensor_desc_t;

// A coalescing free-list node describing an unused arena region.
typedef struct emunpu_free_block_t {
  uint64_t offset;
  uint64_t size;
} emunpu_free_block_t;

#define IREE_EMUNPU_MAX_TENSORS 1024
#define IREE_EMUNPU_MAX_FREE_BLOCKS 1024

typedef struct emunpu_global_mem_t {
  uint8_t* arena;
  uint64_t capacity;
  uint64_t watermark;  // bump pointer past the highest live/free region
  uint64_t bytes_in_use;

  emunpu_tensor_desc_t descs[IREE_EMUNPU_MAX_TENSORS];
  uint32_t desc_count;
  emunpu_tensor_id_t next_id;

  emunpu_free_block_t free_blocks[IREE_EMUNPU_MAX_FREE_BLOCKS];
  uint32_t free_count;
} emunpu_global_mem_t;

// Initializes |gm| with a freshly allocated arena of |capacity| bytes (or
// IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES if 0).
EMUNPU_API emunpu_status_t emunpu_global_mem_initialize(emunpu_global_mem_t* gm,
                                             uint64_t capacity);
EMUNPU_API void emunpu_global_mem_deinitialize(emunpu_global_mem_t* gm);

// Allocates a tensor of the given dtype/shape (rank may be 0 for a raw byte
// buffer, in which case shape[0] is the element count). Returns its id and,
// optionally, its host pointer / descriptor. |name| may be NULL.
EMUNPU_API emunpu_status_t emunpu_global_mem_alloc(emunpu_global_mem_t* gm,
                                        emunpu_dtype_t dtype, uint32_t rank,
                                        const uint32_t* shape, const char* name,
                                        emunpu_tensor_id_t* out_id);

// Allocates a raw buffer of |byte_size| bytes (dtype recorded as u8).
EMUNPU_API emunpu_status_t emunpu_global_mem_alloc_bytes(emunpu_global_mem_t* gm,
                                              uint64_t byte_size,
                                              const char* name,
                                              emunpu_tensor_id_t* out_id);

// Frees a previously allocated tensor, returning its region to the free list.
EMUNPU_API emunpu_status_t emunpu_global_mem_free(emunpu_global_mem_t* gm,
                                       emunpu_tensor_id_t id);

// Looks up the descriptor for |id| (NULL if not found).
EMUNPU_API const emunpu_tensor_desc_t* emunpu_global_mem_lookup(
    const emunpu_global_mem_t* gm, emunpu_tensor_id_t id);

// Host pointer to the start of |id|'s region (NULL if not found).
EMUNPU_API void* emunpu_global_mem_host_ptr(emunpu_global_mem_t* gm,
                                 emunpu_tensor_id_t id);

// Host pointer to an absolute arena byte |offset| (used by core DMA against the
// global address space). NULL if out of range.
EMUNPU_API void* emunpu_global_mem_at(emunpu_global_mem_t* gm, uint64_t offset,
                           uint64_t required_bytes);

// Byte offset of |id| within the arena (the global "address"). Returns
// EMUNPU_ERROR_NOT_FOUND if the id is unknown.
EMUNPU_API emunpu_status_t emunpu_global_mem_offset(const emunpu_global_mem_t* gm,
                                         emunpu_tensor_id_t id,
                                         uint64_t* out_offset);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_GLOBAL_MEM_H_
