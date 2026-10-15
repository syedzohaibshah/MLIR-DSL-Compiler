// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/emunpu/emunpu_mem_fb.h"

#include <stdlib.h>
#include <string.h>

// flatcc-generated readers/builders/verifiers for emunpu_mem.fbs. The schema's
// `namespace iree.hal.emunpu` yields the iree_hal_emunpu_* symbol prefix.
#include "iree/hal/drivers/emunpu/emunpu_mem_builder.h"
#include "iree/hal/drivers/emunpu/emunpu_mem_reader.h"
#include "iree/hal/drivers/emunpu/emunpu_mem_verifier.h"

emunpu_status_t emunpu_mem_fb_serialize(const emunpu_global_mem_t* gm,
                                        void** out_buffer, size_t* out_size) {
  if (!gm || !out_buffer || !out_size) return EMUNPU_ERROR_INVALID_ARGUMENT;
  *out_buffer = NULL;
  *out_size = 0;

  flatcc_builder_t builder;
  flatcc_builder_init(&builder);

  // Build the vector of tensor descriptors.
  iree_hal_emunpu_TensorDescDef_vec_start(&builder);
  for (uint32_t i = 0; i < gm->desc_count; ++i) {
    const emunpu_tensor_desc_t* d = &gm->descs[i];
    iree_hal_emunpu_TensorDescDef_start(&builder);
    iree_hal_emunpu_TensorDescDef_id_add(&builder, d->id);
    iree_hal_emunpu_TensorDescDef_dtype_add(&builder, (uint32_t)d->dtype);
    iree_hal_emunpu_TensorDescDef_shape_create(&builder, d->shape, d->rank);
    iree_hal_emunpu_TensorDescDef_offset_add(&builder, d->offset);
    iree_hal_emunpu_TensorDescDef_size_add(&builder, d->size);
    iree_hal_emunpu_TensorDescDef_name_create_str(&builder, d->name);
    iree_hal_emunpu_TensorDescDef_vec_push(
        &builder, iree_hal_emunpu_TensorDescDef_end(&builder));
  }
  iree_hal_emunpu_TensorDescDef_vec_ref_t tensors =
      iree_hal_emunpu_TensorDescDef_vec_end(&builder);

  iree_hal_emunpu_GlobalMemoryLayoutDef_start_as_root(&builder);
  iree_hal_emunpu_GlobalMemoryLayoutDef_capacity_add(&builder, gm->capacity);
  iree_hal_emunpu_GlobalMemoryLayoutDef_watermark_add(&builder, gm->watermark);
  iree_hal_emunpu_GlobalMemoryLayoutDef_bytes_in_use_add(&builder,
                                                         gm->bytes_in_use);
  iree_hal_emunpu_GlobalMemoryLayoutDef_tensors_add(&builder, tensors);
  iree_hal_emunpu_GlobalMemoryLayoutDef_end_as_root(&builder);

  size_t size = 0;
  void* buffer = flatcc_builder_finalize_buffer(&builder, &size);
  flatcc_builder_clear(&builder);
  if (!buffer) return EMUNPU_ERROR_OUT_OF_MEMORY;
  *out_buffer = buffer;
  *out_size = size;
  return EMUNPU_OK;
}

void emunpu_mem_fb_free(void* buffer) { free(buffer); }

emunpu_status_t emunpu_mem_fb_parse(emunpu_global_mem_t* gm, const void* buffer,
                                    size_t size) {
  if (!gm || !buffer) return EMUNPU_ERROR_INVALID_ARGUMENT;
  if (iree_hal_emunpu_GlobalMemoryLayoutDef_verify_as_root(buffer, size)) {
    return EMUNPU_ERROR_INVALID_ARGUMENT;  // failed verification
  }
  iree_hal_emunpu_GlobalMemoryLayoutDef_table_t root =
      iree_hal_emunpu_GlobalMemoryLayoutDef_as_root(buffer);
  uint64_t capacity = iree_hal_emunpu_GlobalMemoryLayoutDef_capacity(root);
  if (capacity > gm->capacity) return EMUNPU_ERROR_OUT_OF_RANGE;

  // Reset the live table and re-create descriptors at their recorded offsets.
  gm->desc_count = 0;
  gm->free_count = 0;
  gm->bytes_in_use = 0;
  gm->watermark = iree_hal_emunpu_GlobalMemoryLayoutDef_watermark(root);

  iree_hal_emunpu_TensorDescDef_vec_t tensors =
      iree_hal_emunpu_GlobalMemoryLayoutDef_tensors(root);
  size_t n = iree_hal_emunpu_TensorDescDef_vec_len(tensors);
  emunpu_tensor_id_t max_id = 0;
  for (size_t i = 0; i < n; ++i) {
    if (gm->desc_count >= IREE_EMUNPU_MAX_TENSORS) {
      return EMUNPU_ERROR_OUT_OF_MEMORY;
    }
    iree_hal_emunpu_TensorDescDef_table_t t =
        iree_hal_emunpu_TensorDescDef_vec_at(tensors, i);
    emunpu_tensor_desc_t* d = &gm->descs[gm->desc_count++];
    memset(d, 0, sizeof(*d));
    d->id = iree_hal_emunpu_TensorDescDef_id(t);
    d->dtype = (emunpu_dtype_t)iree_hal_emunpu_TensorDescDef_dtype(t);
    d->offset = iree_hal_emunpu_TensorDescDef_offset(t);
    d->size = iree_hal_emunpu_TensorDescDef_size(t);
    flatbuffers_uint32_vec_t shape = iree_hal_emunpu_TensorDescDef_shape(t);
    d->rank = (uint32_t)flatbuffers_uint32_vec_len(shape);
    if (d->rank > IREE_EMUNPU_MAX_RANK) d->rank = IREE_EMUNPU_MAX_RANK;
    for (uint32_t r = 0; r < d->rank; ++r) {
      d->shape[r] = flatbuffers_uint32_vec_at(shape, r);
    }
    flatbuffers_string_t name = iree_hal_emunpu_TensorDescDef_name(t);
    if (name) {
      strncpy(d->name, name, sizeof(d->name) - 1);
      d->name[sizeof(d->name) - 1] = '\0';
    }
    gm->bytes_in_use += d->size;
    if (d->id > max_id) max_id = d->id;
  }
  gm->next_id = max_id + 1;
  return EMUNPU_OK;
}
