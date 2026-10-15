// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Emulated NPU "hardware API".
//
// This is the software model of the emulated accelerator's compute engine. It
// is deliberately isolated from all IREE HAL plumbing: it takes a single coarse
// NPU command plus the raw buffer pointers/lengths for the dispatch bindings
// and performs the computation on the host. Swap this file for a cycle-accurate
// model, an out-of-process simulator, or real device pokes without touching the
// HAL driver wiring.
//===----------------------------------------------------------------------===//

#ifndef IREE_HAL_DRIVERS_EMUNPU_EMUNPU_HW_H_
#define IREE_HAL_DRIVERS_EMUNPU_EMUNPU_HW_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/emunpu/emunpu_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executes a single coarse NPU |command| against the dispatch |bindings|.
//
// |bindings| / |binding_lengths| are the dense per-dispatch binding tables
// exactly as provided in iree_hal_executable_dispatch_state_v0_t; the command's
// operand_bindings/result_binding index into them. |binding_count| bounds the
// table for validation.
//
// |tiles| is the executable's matmul tile array (the compile-time tile
// schedule); a MATMUL command replays tiles[first_tile .. first_tile+tile_count)
// rather than tiling at runtime. May be NULL only if no command uses it.
iree_status_t iree_hal_emunpu_hw_execute_command(
    const iree_hal_emunpu_command_t* command,
    const iree_hal_emunpu_matmul_tile_t* tiles, iree_host_size_t binding_count,
    void* const* bindings, const iree_host_size_t* binding_lengths);

// Returns true when execution tracing is enabled (env IREE_EMUNPU_TRACE != 0).
// When enabled, callers should log each command via the function below before
// executing it; this is opt-in proof that a workload ran on the emulated NPU.
bool iree_hal_emunpu_hw_trace_enabled(void);

// Logs a single coarse |command| to stderr with a running op counter. |command|
// should reference operands/results in the ISA encoding (bindings as bN, scratch
// slots as sN) so the trace reflects the program as compiled.
void iree_hal_emunpu_hw_trace_command(
    const iree_hal_emunpu_command_t* command);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_EMUNPU_EMUNPU_HW_H_
