// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/emunpu/emunpu_loader.h"

#include <stdbool.h>
#include <string.h>

#include "iree/hal/api.h"
#include "iree/hal/drivers/emunpu/emunpu_hw.h"
#include "iree/hal/drivers/emunpu/emunpu_isa.h"
#include "iree/hal/local/local_executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_emunpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_emunpu_executable_t {
  iree_hal_local_executable_t base;

  // Owned copy of the command blob; all pointers below alias into it.
  uint8_t* blob;
  iree_host_size_t blob_size;

  // Parsed views into |blob|.
  const iree_hal_emunpu_header_t* header;
  const iree_hal_emunpu_export_t* exports;
  const iree_hal_emunpu_command_t* commands;
  const iree_hal_emunpu_matmul_tile_t* tiles;
  const char* string_table;

  // Per-export dispatch attrs (binding/constant counts) consumed by the inline
  // command buffer for validation. Owned; sized header->export_count.
  iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs;
} iree_hal_emunpu_executable_t;

static const iree_hal_local_executable_vtable_t
    iree_hal_emunpu_executable_vtable;

// Validates the structural integrity of a command blob.
static iree_status_t iree_hal_emunpu_executable_verify_blob(
    iree_const_byte_span_t data) {
  if (data.data_length < sizeof(iree_hal_emunpu_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "emunpu blob too small for header");
  }
  const iree_hal_emunpu_header_t* header =
      (const iree_hal_emunpu_header_t*)data.data;
  if (header->magic != IREE_HAL_EMUNPU_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "emunpu blob magic mismatch (got %08X)",
                            header->magic);
  }
  if (header->version != IREE_HAL_EMUNPU_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "emunpu blob version %u unsupported (expected %u)",
                            header->version, IREE_HAL_EMUNPU_VERSION);
  }
  const iree_host_size_t commands_offset =
      iree_hal_emunpu_command_table_offset(header->export_count);
  const iree_host_size_t commands_end =
      commands_offset + (iree_host_size_t)header->command_count *
                            sizeof(iree_hal_emunpu_command_t);
  const iree_host_size_t tiles_end =
      iree_hal_emunpu_tile_table_offset(header->export_count,
                                        header->command_count) +
      (iree_host_size_t)header->tile_count *
          sizeof(iree_hal_emunpu_matmul_tile_t);
  if (commands_end > data.data_length || tiles_end > data.data_length ||
      (iree_host_size_t)header->string_table_offset +
              header->string_table_size >
          data.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "emunpu blob tables exceed blob bounds");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_emunpu_executable_create(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_emunpu_executable_verify_blob(
                                        executable_params->executable_data));
  const iree_hal_emunpu_header_t* src_header =
      (const iree_hal_emunpu_header_t*)executable_params->executable_data.data;

  // Allocate the executable plus an owned copy of the blob and the dispatch
  // attrs array in a single allocation.
  iree_hal_emunpu_executable_t* executable = NULL;
  const iree_host_size_t attrs_size =
      src_header->export_count *
      sizeof(iree_hal_executable_dispatch_attrs_v0_t);
  const iree_host_size_t blob_size =
      executable_params->executable_data.data_length;
  const iree_host_size_t total_size =
      sizeof(*executable) + attrs_size + blob_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_local_executable_initialize(&iree_hal_emunpu_executable_vtable,
                                       host_allocator, &executable->base);

  executable->dispatch_attrs =
      (iree_hal_executable_dispatch_attrs_v0_t*)((uint8_t*)executable +
                                                 sizeof(*executable));
  executable->blob = (uint8_t*)executable->dispatch_attrs + attrs_size;
  executable->blob_size = blob_size;
  memcpy(executable->blob, executable_params->executable_data.data, blob_size);

  executable->header = (const iree_hal_emunpu_header_t*)executable->blob;
  executable->exports =
      (const iree_hal_emunpu_export_t*)(executable->blob +
                                        iree_hal_emunpu_export_table_offset());
  executable->commands =
      (const iree_hal_emunpu_command_t*)(executable->blob +
                                         iree_hal_emunpu_command_table_offset(
                                             executable->header->export_count));
  executable->tiles =
      (const iree_hal_emunpu_matmul_tile_t*)(executable->blob +
                                             iree_hal_emunpu_tile_table_offset(
                                                 executable->header
                                                     ->export_count,
                                                 executable->header
                                                     ->command_count));
  executable->string_table =
      (const char*)(executable->blob + executable->header->string_table_offset);

  for (uint32_t i = 0; i < executable->header->export_count; ++i) {
    iree_hal_executable_dispatch_attrs_v0_t attrs = {0};
    attrs.binding_count = (uint8_t)executable->exports[i].binding_count;
    attrs.constant_count = (uint8_t)executable->exports[i].constant_count;
    executable->dispatch_attrs[i] = attrs;
  }
  executable->base.dispatch_attrs = executable->dispatch_attrs;
  // No native function pointers: dispatches are interpreted in issue_call.
  executable->base.dispatch_ptrs = NULL;

  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_emunpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_emunpu_executable_t* executable =
      (iree_hal_emunpu_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

// Maximum scratch slots a single (fused) dispatch may use. A small fixed cap
// keeps the per-call setup on the stack and bounds untrusted blob inputs.
#define IREE_HAL_EMUNPU_MAX_SCRATCH 8

static iree_host_size_t iree_hal_emunpu_dtype_size_local(
    iree_hal_emunpu_dtype_t dtype) {
  switch (dtype) {
    case IREE_HAL_EMUNPU_DTYPE_F32:
      return sizeof(float);
    case IREE_HAL_EMUNPU_DTYPE_I32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}

// Byte size a command writes to its result (used to size scratch buffers).
static iree_host_size_t iree_hal_emunpu_command_result_bytes(
    const iree_hal_emunpu_command_t* command) {
  const iree_host_size_t elem =
      iree_hal_emunpu_dtype_size_local(command->dtype);
  if (command->opcode == IREE_HAL_EMUNPU_OP_MATMUL) {
    return (iree_host_size_t)command->dims[0] * command->dims[1] * elem;
  }
  return (iree_host_size_t)command->dims[0] * elem;
}

static iree_status_t iree_hal_emunpu_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_emunpu_executable_t* executable =
      (iree_hal_emunpu_executable_t*)base_executable;
  if (IREE_UNLIKELY(ordinal >= executable->header->export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export ordinal %" PRIhsz " out of bounds",
                            ordinal);
  }

  // The emulated NPU executes each coarse op exactly once over the whole
  // tensor. The dispatch is recorded with a single workgroup, but guard
  // regardless so a larger workgroup count never re-runs (and corrupts) the
  // computation.
  if (workgroup_state->workgroup_id_x != 0 ||
      workgroup_state->workgroup_id_y != 0 ||
      workgroup_state->workgroup_id_z != 0) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_hal_emunpu_export_t* export = &executable->exports[ordinal];
  const iree_hal_emunpu_command_t* cmds =
      &executable->commands[export->first_command];
  const iree_host_size_t binding_count = dispatch_state->binding_count;

  // Scan the command range to size the scratch slots that carry intermediate
  // results between fused ops. A slot's size comes from the command that writes
  // it; references are validated against the cap.
  iree_host_size_t scratch_bytes[IREE_HAL_EMUNPU_MAX_SCRATCH] = {0};
  uint32_t scratch_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < export->command_count && iree_status_is_ok(status);
       ++i) {
    const int32_t refs[4] = {cmds[i].operand_bindings[0],
                             cmds[i].operand_bindings[1],
                             cmds[i].operand_bindings[2],
                             cmds[i].result_binding};
    for (int r = 0; r < 4; ++r) {
      if (!IREE_HAL_EMUNPU_REF_IS_SCRATCH(refs[r])) continue;
      uint32_t slot = IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(refs[r]);
      if (slot >= IREE_HAL_EMUNPU_MAX_SCRATCH) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "scratch slot %u exceeds cap %d", slot,
                                  IREE_HAL_EMUNPU_MAX_SCRATCH);
        break;
      }
      if (slot + 1 > scratch_count) scratch_count = slot + 1;
    }
    if (IREE_HAL_EMUNPU_REF_IS_SCRATCH(cmds[i].result_binding)) {
      scratch_bytes[IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(cmds[i].result_binding)] =
          iree_hal_emunpu_command_result_bytes(&cmds[i]);
    }
  }

  // Build merged pointer/length arrays: [0, binding_count) are the HAL
  // bindings, [binding_count, binding_count + scratch_count) are scratch.
  const iree_host_size_t total = binding_count + scratch_count;
  void* ptrs[IREE_HAL_EMUNPU_MAX_SCRATCH + 32];
  iree_host_size_t lengths[IREE_HAL_EMUNPU_MAX_SCRATCH + 32];
  void* scratch_ptrs[IREE_HAL_EMUNPU_MAX_SCRATCH] = {0};
  iree_allocator_t host_allocator = executable->base.host_allocator;
  if (iree_status_is_ok(status) &&
      total > IREE_ARRAYSIZE(ptrs)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "dispatch needs %" PRIhsz
                              " buffers, exceeding the emunpu limit",
                              total);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      ptrs[i] = dispatch_state->binding_ptrs[i];
      lengths[i] = dispatch_state->binding_lengths[i];
    }
    for (uint32_t s = 0; s < scratch_count && iree_status_is_ok(status); ++s) {
      status = iree_allocator_malloc(host_allocator, scratch_bytes[s],
                                     &scratch_ptrs[s]);
      ptrs[binding_count + s] = scratch_ptrs[s];
      lengths[binding_count + s] = scratch_bytes[s];
    }
  }

  // Execute, remapping scratch refs to their merged-array ordinal. Trace the
  // original (pre-remap) command so scratch slots show as sN, not merged bN.
  const bool trace = iree_hal_emunpu_hw_trace_enabled();
  for (uint32_t i = 0; i < export->command_count && iree_status_is_ok(status);
       ++i) {
    if (trace) iree_hal_emunpu_hw_trace_command(&cmds[i]);
    iree_hal_emunpu_command_t local = cmds[i];
    int32_t* slots[4] = {&local.operand_bindings[0], &local.operand_bindings[1],
                         &local.operand_bindings[2], &local.result_binding};
    for (int r = 0; r < 4; ++r) {
      if (IREE_HAL_EMUNPU_REF_IS_SCRATCH(*slots[r])) {
        *slots[r] = (int32_t)(binding_count +
                              IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(*slots[r]));
      }
    }
    status = iree_hal_emunpu_hw_execute_command(&local, executable->tiles,
                                                total, ptrs, lengths);
  }

  for (uint32_t s = 0; s < scratch_count; ++s) {
    iree_allocator_free(host_allocator, scratch_ptrs[s]);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_host_size_t iree_hal_emunpu_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_emunpu_executable_t* executable =
      (iree_hal_emunpu_executable_t*)base_executable;
  return executable->header->export_count;
}

static iree_status_t iree_hal_emunpu_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_emunpu_executable_t* executable =
      (iree_hal_emunpu_executable_t*)base_executable;
  if (export_ordinal >= executable->header->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal out of bounds");
  }
  const iree_hal_emunpu_export_t* export = &executable->exports[export_ordinal];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = iree_make_string_view(
      executable->string_table + export->name_offset, export->name_length);
  out_info->constant_count = (uint16_t) export->constant_count;
  out_info->binding_count = (uint16_t) export->binding_count;
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_emunpu_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  // Reflection of individual parameters is not provided by this backend.
  return iree_ok_status();
}

static iree_status_t iree_hal_emunpu_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_emunpu_executable_t* executable =
      (iree_hal_emunpu_executable_t*)base_executable;
  for (uint32_t i = 0; i < executable->header->export_count; ++i) {
    const iree_hal_emunpu_export_t* export = &executable->exports[i];
    iree_string_view_t export_name = iree_make_string_view(
        executable->string_table + export->name_offset, export->name_length);
    if (iree_string_view_equal(name, export_name)) {
      *out_export_ordinal = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "export '%.*s' not found",
                          (int)name.size, name.data);
}

static const iree_hal_local_executable_vtable_t
    iree_hal_emunpu_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_emunpu_executable_destroy,
                .export_count = iree_hal_emunpu_executable_export_count,
                .export_info = iree_hal_emunpu_executable_export_info,
                .export_parameters =
                    iree_hal_emunpu_executable_export_parameters,
                .lookup_export_by_name =
                    iree_hal_emunpu_executable_lookup_export_by_name,
            },
        .issue_call = iree_hal_emunpu_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_emunpu_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_emunpu_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_emunpu_loader_t;

static const iree_hal_executable_loader_vtable_t iree_hal_emunpu_loader_vtable;

iree_status_t iree_hal_emunpu_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_emunpu_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_emunpu_loader_vtable,
        iree_hal_executable_import_provider_null(), &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_emunpu_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_emunpu_loader_t* executable_loader =
      (iree_hal_emunpu_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(host_allocator, executable_loader);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_emunpu_loader_infer_format(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "emunpu loader requires an explicit format");
}

static bool iree_hal_emunpu_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(
      executable_format,
      iree_make_cstring_view(IREE_HAL_EMUNPU_EXECUTABLE_FORMAT));
}

static iree_status_t iree_hal_emunpu_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_emunpu_loader_t* executable_loader =
      (iree_hal_emunpu_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_emunpu_executable_create(
      executable_params, executable_loader->host_allocator, out_executable);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t iree_hal_emunpu_loader_vtable =
    {
        .destroy = iree_hal_emunpu_loader_destroy,
        .infer_format = iree_hal_emunpu_loader_infer_format,
        .query_support = iree_hal_emunpu_loader_query_support,
        .try_load = iree_hal_emunpu_loader_try_load,
};
