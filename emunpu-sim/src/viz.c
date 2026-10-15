// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emunpu/viz.h"

#include <stdio.h>
#include <stdlib.h>

// 0 = unknown (not yet read), 1 = enabled, 2 = disabled.
static int g_viz_state = 0;
static FILE* g_viz_file = NULL;

int emunpu_viz_enabled(void) {
  if (g_viz_state == 0) {
    const char* p = getenv("IREE_EMUNPU_VIZ");
    g_viz_state = (p && p[0] != '\0') ? 1 : 2;
  }
  return g_viz_state == 1;
}

// Returns the (lazily opened) event file, or NULL if visualization is off. The
// file is truncated the first time it is opened in a process so each run starts
// clean; the Python viewer also resets its state on the "RUN" marker below.
static FILE* emunpu_viz_file(void) {
  if (!emunpu_viz_enabled()) return NULL;
  if (!g_viz_file) {
    const char* p = getenv("IREE_EMUNPU_VIZ");
    g_viz_file = fopen(p, "w");
    if (g_viz_file) setvbuf(g_viz_file, NULL, _IOLBF, 0);
  }
  return g_viz_file;
}

// Sub-operation name for the ops that have one ("EW.add", "ACT.gelu", ...), or
// NULL when the opcode alone says everything.
static const char* emunpu_viz_subname(const emunpu_instr_t* in) {
  switch (in->op) {
    case EMUNPU_I_EW:
      return emunpu_ew_name((emunpu_ew_kind_t)in->flags);
    case EMUNPU_I_REDUCE:
      return emunpu_reduce_name((emunpu_reduce_kind_t)in->flags);
    case EMUNPU_I_ACT:
      return emunpu_act_name((emunpu_act_kind_t)in->flags);
    case EMUNPU_I_NORM:
      return emunpu_norm_name((emunpu_norm_kind_t)in->flags);
    case EMUNPU_I_POOL2D:
      return emunpu_pool_name((emunpu_pool_kind_t)in->flags);
    case EMUNPU_I_SOFTMAX:
      return (in->flags & EMUNPU_SOFTMAX_LOG) ? "log" : NULL;
    default:
      return NULL;
  }
}

void emunpu_viz_run_begin(uint32_t num_cores, uint64_t global_capacity) {
  FILE* f = emunpu_viz_file();
  if (!f) return;
  fprintf(f, "RUN cores=%u cap=%llu\n", num_cores,
          (unsigned long long)global_capacity);
  fflush(f);
}

void emunpu_viz_gemm(uint32_t M, uint32_t N, uint32_t K, int transpose_b,
                     uint32_t tiles, uint32_t cores) {
  FILE* f = emunpu_viz_file();
  if (!f) return;
  fprintf(f, "GEMM m=%u n=%u k=%u t=%d tiles=%u cores=%u\n", M, N, K,
          transpose_b ? 1 : 0, tiles, cores);
  fflush(f);
}

void emunpu_viz_instr(uint32_t core_id, const emunpu_instr_t* in) {
  FILE* f = emunpu_viz_file();
  if (!f || !in) return;
  // |kind| stays the bare opcode mnemonic (the viewer buckets on it); the
  // sub-operation, when there is one, rides along in an optional |sub| field.
  const char* sub = emunpu_viz_subname(in);
  fprintf(f, "INSTR core=%u kind=%s dt=%s m=%u n=%u k=%u f=%u", core_id,
          emunpu_iop_name(in->op), emunpu_dtype_name(in->c_dtype), in->M, in->N,
          in->K, in->flags);
  if (sub) fprintf(f, " sub=%s", sub);
  fputc('\n', f);
  fflush(f);
}

void emunpu_viz_mem(uint64_t used, uint64_t capacity) {
  FILE* f = emunpu_viz_file();
  if (!f) return;
  fprintf(f, "MEM used=%llu cap=%llu\n", (unsigned long long)used,
          (unsigned long long)capacity);
  fflush(f);
}

void emunpu_viz_barrier(uint32_t streams) {
  FILE* f = emunpu_viz_file();
  if (!f) return;
  fprintf(f, "BARRIER streams=%u\n", streams);
  fflush(f);
}
