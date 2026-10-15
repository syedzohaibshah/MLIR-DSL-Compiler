// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emunpu/global_mem.h"

#include <stdlib.h>
#include <string.h>

// All allocations are aligned to this many bytes so 16x16 tiles of any dtype
// stay naturally aligned for the cores' loads/stores.
#define IREE_EMUNPU_GLOBAL_ALIGNMENT 64

static uint64_t emunpu_align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

emunpu_status_t emunpu_global_mem_initialize(emunpu_global_mem_t* gm,
                                             uint64_t capacity) {
  if (!gm) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (capacity == 0) capacity = IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES;
  memset(gm, 0, sizeof(*gm));
  gm->arena = (uint8_t*)malloc((size_t)capacity);
  if (!gm->arena) return EMUNPU_ERROR_OUT_OF_MEMORY;
  gm->capacity = capacity;
  gm->next_id = 1;  // 0 is reserved as the null id
  return EMUNPU_OK;
}

void emunpu_global_mem_deinitialize(emunpu_global_mem_t* gm) {
  if (!gm) return;
  free(gm->arena);
  memset(gm, 0, sizeof(*gm));
}

// Removes free block |idx| by swapping with the last entry.
static void emunpu_free_list_remove(emunpu_global_mem_t* gm, uint32_t idx) {
  gm->free_blocks[idx] = gm->free_blocks[--gm->free_count];
}

// Carves |size| aligned bytes out of the arena, preferring to reuse a free
// block (first fit) before bumping the watermark. Returns the byte offset.
static emunpu_status_t emunpu_arena_carve(emunpu_global_mem_t* gm,
                                          uint64_t size, uint64_t* out_offset) {
  size = emunpu_align_up(size, IREE_EMUNPU_GLOBAL_ALIGNMENT);
  // First fit over the free list.
  for (uint32_t i = 0; i < gm->free_count; ++i) {
    if (gm->free_blocks[i].size >= size) {
      *out_offset = gm->free_blocks[i].offset;
      if (gm->free_blocks[i].size == size) {
        emunpu_free_list_remove(gm, i);
      } else {
        gm->free_blocks[i].offset += size;
        gm->free_blocks[i].size -= size;
      }
      gm->bytes_in_use += size;
      return EMUNPU_OK;
    }
  }
  // Otherwise bump the watermark.
  if (gm->watermark + size > gm->capacity) {
    return EMUNPU_ERROR_OUT_OF_MEMORY;
  }
  *out_offset = gm->watermark;
  gm->watermark += size;
  gm->bytes_in_use += size;
  return EMUNPU_OK;
}

// Returns a freed region to the free list, coalescing with adjacent blocks.
static void emunpu_arena_release(emunpu_global_mem_t* gm, uint64_t offset,
                                 uint64_t size) {
  size = emunpu_align_up(size, IREE_EMUNPU_GLOBAL_ALIGNMENT);
  gm->bytes_in_use -= size;
  // Coalesce with neighbors already in the free list.
  for (uint32_t i = 0; i < gm->free_count;) {
    emunpu_free_block_t* b = &gm->free_blocks[i];
    if (b->offset + b->size == offset) {  // block precedes us
      offset = b->offset;
      size += b->size;
      emunpu_free_list_remove(gm, i);
      continue;
    }
    if (offset + size == b->offset) {  // block follows us
      size += b->size;
      emunpu_free_list_remove(gm, i);
      continue;
    }
    ++i;
  }
  // If the merged region now reaches the watermark, just lower it.
  if (offset + size == gm->watermark) {
    gm->watermark = offset;
    return;
  }
  if (gm->free_count < IREE_EMUNPU_MAX_FREE_BLOCKS) {
    gm->free_blocks[gm->free_count].offset = offset;
    gm->free_blocks[gm->free_count].size = size;
    gm->free_count++;
  }
  // If the free list is full we simply leak the region until reset; bounded and
  // benign for an emulator.
}

emunpu_status_t emunpu_global_mem_alloc(emunpu_global_mem_t* gm,
                                        emunpu_dtype_t dtype, uint32_t rank,
                                        const uint32_t* shape, const char* name,
                                        emunpu_tensor_id_t* out_id) {
  if (!gm || !out_id) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (rank > IREE_EMUNPU_MAX_RANK) return EMUNPU_ERROR_INVALID_ARGUMENT;
  size_t elem_size = emunpu_dtype_size(dtype);
  if (elem_size == 0) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (gm->desc_count >= IREE_EMUNPU_MAX_TENSORS) {
    return EMUNPU_ERROR_OUT_OF_MEMORY;
  }

  uint64_t elems = 1;
  for (uint32_t i = 0; i < rank; ++i) elems *= shape ? shape[i] : 1;
  if (rank == 0 && shape) elems = shape[0];  // raw element count
  uint64_t size = elems * elem_size;

  uint64_t offset = 0;
  emunpu_status_t status = emunpu_arena_carve(gm, size, &offset);
  if (status != EMUNPU_OK) return status;

  emunpu_tensor_desc_t* d = &gm->descs[gm->desc_count++];
  memset(d, 0, sizeof(*d));
  d->id = gm->next_id++;
  d->dtype = dtype;
  d->rank = rank;
  for (uint32_t i = 0; i < rank; ++i) d->shape[i] = shape ? shape[i] : 1;
  d->offset = offset;
  d->size = size;
  if (name) {
    strncpy(d->name, name, sizeof(d->name) - 1);
    d->name[sizeof(d->name) - 1] = '\0';
  }
  *out_id = d->id;
  return EMUNPU_OK;
}

emunpu_status_t emunpu_global_mem_alloc_bytes(emunpu_global_mem_t* gm,
                                              uint64_t byte_size,
                                              const char* name,
                                              emunpu_tensor_id_t* out_id) {
  uint32_t shape = (uint32_t)byte_size;
  return emunpu_global_mem_alloc(gm, EMUNPU_DT_UINT8, 0, &shape, name, out_id);
}

static int emunpu_find_desc(const emunpu_global_mem_t* gm,
                            emunpu_tensor_id_t id) {
  for (uint32_t i = 0; i < gm->desc_count; ++i) {
    if (gm->descs[i].id == id) return (int)i;
  }
  return -1;
}

emunpu_status_t emunpu_global_mem_free(emunpu_global_mem_t* gm,
                                       emunpu_tensor_id_t id) {
  if (!gm) return EMUNPU_ERROR_INVALID_ARGUMENT;
  int idx = emunpu_find_desc(gm, id);
  if (idx < 0) return EMUNPU_ERROR_NOT_FOUND;
  emunpu_arena_release(gm, gm->descs[idx].offset, gm->descs[idx].size);
  gm->descs[idx] = gm->descs[--gm->desc_count];
  return EMUNPU_OK;
}

const emunpu_tensor_desc_t* emunpu_global_mem_lookup(
    const emunpu_global_mem_t* gm, emunpu_tensor_id_t id) {
  if (!gm) return NULL;
  int idx = emunpu_find_desc(gm, id);
  return idx < 0 ? NULL : &gm->descs[idx];
}

void* emunpu_global_mem_host_ptr(emunpu_global_mem_t* gm,
                                 emunpu_tensor_id_t id) {
  const emunpu_tensor_desc_t* d = emunpu_global_mem_lookup(gm, id);
  return d ? gm->arena + d->offset : NULL;
}

void* emunpu_global_mem_at(emunpu_global_mem_t* gm, uint64_t offset,
                           uint64_t required_bytes) {
  if (!gm || offset + required_bytes > gm->capacity) return NULL;
  return gm->arena + offset;
}

emunpu_status_t emunpu_global_mem_offset(const emunpu_global_mem_t* gm,
                                         emunpu_tensor_id_t id,
                                         uint64_t* out_offset) {
  const emunpu_tensor_desc_t* d = emunpu_global_mem_lookup(gm, id);
  if (!d) return EMUNPU_ERROR_NOT_FOUND;
  *out_offset = d->offset;
  return EMUNPU_OK;
}
