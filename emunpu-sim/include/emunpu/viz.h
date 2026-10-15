// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Visualization event stream.
//
// When the environment variable IREE_EMUNPU_VIZ names a file path, the engine
// appends a machine-readable, line-oriented event stream describing device
// activity (run start, GEMM tiling, every per-core instruction, global memory
// usage, barriers). A separate terminal tool (`emunpu_demos/emunpu_viz.py`) tails that file
// and renders a live dashboard of the cores while a workload runs.
//
// Emission is a no-op (a single cached env check) when the variable is unset,
// so it adds nothing to ordinary runs. IREE-independent (stdio only).
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_VIZ_H_
#define EMUNPU_VIZ_H_

#include "emunpu/arch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// True when IREE_EMUNPU_VIZ is set (cached after the first call).
EMUNPU_API int emunpu_viz_enabled(void);

// Emits a "new run" marker; truncates the event file on first use of a process.
EMUNPU_API void emunpu_viz_run_begin(uint32_t num_cores, uint64_t global_capacity);

// Emits a high-level GEMM event (the scheduler about to fan tiles out).
EMUNPU_API void emunpu_viz_gemm(uint32_t M, uint32_t N, uint32_t K, int transpose_b,
                     uint32_t tiles, uint32_t cores);

// Emits a per-core instruction event (what a core just executed).
EMUNPU_API void emunpu_viz_instr(uint32_t core_id, const emunpu_instr_t* in);

// Emits the current global memory usage.
EMUNPU_API void emunpu_viz_mem(uint64_t used, uint64_t capacity);

// Emits a device barrier event.
EMUNPU_API void emunpu_viz_barrier(uint32_t streams);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EMUNPU_VIZ_H_
