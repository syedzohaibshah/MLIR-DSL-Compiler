// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_EMUNPU_EMUNPU_LOADER_H_
#define IREE_HAL_DRIVERS_EMUNPU_EMUNPU_LOADER_H_

#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an executable loader that loads `emunpu-cmd-fb` command blobs
// produced by the emunpu compiler target backend and executes them on the
// emulated NPU "hardware API". This is the only loader registered with the
// emunpu device.
iree_status_t iree_hal_emunpu_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_EMUNPU_EMUNPU_LOADER_H_
