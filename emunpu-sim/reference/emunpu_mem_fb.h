// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// FlatBuffer (de)serialization of the global memory layout.
//
// Bridges emunpu_global_mem_t <-> the `emunpu_mem.fbs` schema. The serialized
// form is the portable, inspectable record of the device's memory layout; the
// HAL driver can dump it, and a fresh device can re-create an identical arena
// layout from it (parse). Kept separate from emunpu_global_mem.c so the core
// allocator has no flatcc dependency and unit-tests standalone.
//===----------------------------------------------------------------------===//

#ifndef IREE_HAL_DRIVERS_EMUNPU_EMUNPU_MEM_FB_H_
#define IREE_HAL_DRIVERS_EMUNPU_EMUNPU_MEM_FB_H_

#include "iree/hal/drivers/emunpu/emunpu_global_mem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Serializes |gm|'s current layout (capacity, watermark, live descriptors) into
// a freshly malloc'd FlatBuffer. The caller owns *out_buffer and frees it with
// emunpu_mem_fb_free.
emunpu_status_t emunpu_mem_fb_serialize(const emunpu_global_mem_t* gm,
                                        void** out_buffer, size_t* out_size);

// Frees a buffer returned by emunpu_mem_fb_serialize.
void emunpu_mem_fb_free(void* buffer);

// Verifies and parses a serialized layout, rebuilding |gm|'s descriptor table
// to match (the arena is reset and re-carved to the recorded offsets/sizes).
// |gm| must already be initialized with capacity >= the recorded watermark.
emunpu_status_t emunpu_mem_fb_parse(emunpu_global_mem_t* gm, const void* buffer,
                                    size_t size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_EMUNPU_EMUNPU_MEM_FB_H_
