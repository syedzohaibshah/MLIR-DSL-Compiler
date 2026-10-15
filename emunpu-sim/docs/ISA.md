# EmuNPU Executable Format and Instruction Set Specification

**Document status:** Stable
**Format identifier:** `emunpu-cmd-fb`
**Binary version:** 4
**Normative header:** [`reference/emunpu_isa.h`](../reference/emunpu_isa.h)

---

## Revision history

| Version | Summary of changes |
|:-------:|--------------------|
| 1 | Initial single-command dispatch format (whole-tensor coarse ops). |
| 2 | Multi-command (fused) dispatches; scratch-slot operand/result references for intermediate tensors; broadcast period for the elementwise `ADD`. |
| 3 | Per-operand byte offsets (`operand_offsets` / `result_offset`) for sub-buffer addressing. |
| 4 | Compile-time matmul tile schedule (`emunpu_matmul_tile_t` records + `tile_count`); `MATMUL` commands carry `first_tile` / `tile_count`. |

---

## 1. Scope

This document specifies the **EmuNPU executable format** — the binary contract
produced by an EmuNPU compiler backend and consumed by an EmuNPU runtime driver.
It defines the container layout, the coarse-grained instruction set, the operand
addressing model, and the matmul tile-schedule encoding.

The EmuNPU is a coarse-grained accelerator: it executes a small set of
whole-tensor operations (`MATMUL`, `ADD`, `RELU`, `COPY`) rather than scalar
code. A compiled dispatch is therefore a short list of high-level command
records that reference dispatch bindings by ordinal, together with a
compile-time tile schedule for each matmul.

The normative definitions are the C structures in
[`reference/emunpu_isa.h`](../reference/emunpu_isa.h). Where this prose and that
header disagree, the header governs.

> **This is not the whole instruction set of the device.** The blob format is the
> narrow contract a *compiler* emits. The hardware itself executes a larger
> micro-ISA — activations, softmax, layer/RMS normalization, transpose, im2col,
> pooling, gather, and extended reductions — specified in
> [`MICRO_ISA.md`](MICRO_ISA.md) and reached through the device/runtime API. This
> document covers version 4 of the blob, which exposes four coarse ops;
> [Appendix C](#appendix-c--non-normative-proposed-v5-opcodes) sketches how a
> future version would expose the rest.

## 2. Conventions

- **Byte order** is little-endian for all scalar fields.
- **Offsets** are byte offsets measured from the first byte of the blob unless
  stated otherwise.
- All multi-byte scalar fields are naturally aligned; every structure in this
  format is composed exclusively of 32-bit fields and is therefore 4-byte
  aligned with no internal padding.
- The keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used in the
  sense of RFC 2119.
- Field types: `u32` = unsigned 32-bit, `i32` = signed 32-bit (two's complement).

## 3. Architectural parameters

The format is parameterized by two physical device constants. **Both the
producer and the consumer MUST agree on these values**; they are defined in
[`include/emunpu/arch.h`](../include/emunpu/arch.h).

| Constant | Value | Meaning |
|----------|:-----:|---------|
| `IREE_EMUNPU_MMA_DIM`   | 16 | Edge length of the tensor-core MMA tile. Matmuls are blocked into `MMA_DIM × MMA_DIM` output tiles. |
| `IREE_EMUNPU_NUM_CORES` | 10 | Number of compute cores. Every `core_id` in a tile schedule MUST be `< NUM_CORES`. |

## 4. Data types

The element type a command operates on is encoded in the command's `dtype`
field.

| Name | Value | Element | Width |
|------|:-----:|---------|:-----:|
| `IREE_HAL_EMUNPU_DTYPE_F32` | 1 | IEEE-754 binary32 | 4 bytes |
| `IREE_HAL_EMUNPU_DTYPE_I32` | 2 | signed 32-bit int  | 4 bytes |

> The executable-format dtype enumeration is intentionally narrow (`F32`, `I32`).
> The underlying hardware model supports seven element types; those additional
> types are exercised through the device runtime API, not this coarse blob
> format.

## 5. Container format

### 5.1 Overall layout

A blob is five contiguous, densely packed sections in the following fixed order:

```
 offset 0
 ┌──────────────────────────────────────────────┐
 │ Header            (1 × emunpu_header_t)        │  28 bytes
 ├──────────────────────────────────────────────┤
 │ Export table      (export_count × export_t)    │  24 bytes each
 ├──────────────────────────────────────────────┤
 │ Command table     (command_count × command_t)  │  64 bytes each
 ├──────────────────────────────────────────────┤
 │ Tile table        (tile_count × matmul_tile_t) │  32 bytes each
 ├──────────────────────────────────────────────┤
 │ String table      (string_table_size bytes)    │  variable
 └──────────────────────────────────────────────┘
```

Each section begins immediately after the previous one; there is no padding or
alignment gap between sections. The canonical offset helpers are provided as
`static inline` functions in the normative header
(`iree_hal_emunpu_export_table_offset`, `..._command_table_offset`,
`..._tile_table_offset`).

### 5.2 Header — `emunpu_header_t` (28 bytes)

| Offset | Size | Field | Description |
|:------:|:----:|-------|-------------|
| 0  | 4 | `magic`               | MUST equal `0x55504E45` (`"ENPU"`, little-endian). |
| 4  | 4 | `version`             | Format version. This specification describes version `4`. |
| 8  | 4 | `export_count`        | Number of `emunpu_export_t` records. |
| 12 | 4 | `command_count`       | Total number of `emunpu_command_t` records. |
| 16 | 4 | `tile_count`          | Total number of `emunpu_matmul_tile_t` records. |
| 20 | 4 | `string_table_offset` | Byte offset of the string table from blob start. |
| 24 | 4 | `string_table_size`   | Size of the string table in bytes. |

A consumer **MUST** reject a blob whose `magic` or `version` does not match the
values it supports.

## 6. Tables

### 6.1 Export table — `emunpu_export_t` (24 bytes)

One record per exported entry point. The commands belonging to an export are the
contiguous range `commands[first_command .. first_command + command_count)`.

| Offset | Size | Field | Description |
|:------:|:----:|-------|-------------|
| 0  | 4 | `name_offset`    | Byte offset of the export name within the string table. |
| 4  | 4 | `name_length`    | Length of the name in bytes. Names are **not** NUL-terminated. |
| 8  | 4 | `first_command`  | Index of the first command for this export. |
| 12 | 4 | `command_count`  | Number of commands for this export. |
| 16 | 4 | `binding_count`  | Distinct dispatch bindings the export consumes. |
| 20 | 4 | `constant_count` | Number of 32-bit push constants the export consumes. |

### 6.2 String table

A flat byte region holding export names. Names are referenced by
`(name_offset, name_length)` pairs and are **not** NUL-terminated. Consumers
MUST treat `name_offset + name_length` as bounded by `string_table_size`.

## 7. Operand reference encoding

Every command operand and result is an `i32` reference interpreted as follows:

| Value range | Meaning |
|-------------|---------|
| `ref >= 0`  | **Dispatch binding ordinal** — an index into the runtime-supplied binding table. |
| `ref == -1` | **Unused** slot (`IREE_HAL_EMUNPU_REF_UNUSED`). |
| `ref <= -2` | **Scratch slot** `s = (-ref) - 2`. `-2` ⇒ slot 0, `-3` ⇒ slot 1, and so on. |

Scratch slots name dispatch-local intermediate buffers that carry results
between commands of a fused chain (e.g. `MATMUL → ADD → RELU`) without occupying
a HAL binding. The runtime allocates one buffer per scratch slot for the
lifetime of a single dispatch.

Helper macros in the normative header: `IREE_HAL_EMUNPU_REF_UNUSED`,
`IREE_HAL_EMUNPU_REF_IS_SCRATCH(ref)`, `IREE_HAL_EMUNPU_REF_SCRATCH_SLOT(ref)`,
`IREE_HAL_EMUNPU_SCRATCH_REF(slot)`.

### 7.1 Operand byte offsets

Each operand and result additionally carries a `u32` byte offset
(`operand_offsets[i]`, `result_offset`) that is **added to the resolved binding
pointer**. This allows a producer to pack several tensors into one buffer and
address each via a static sub-span offset. Offsets are **ignored for scratch
references**, which always refer to a whole buffer.

## 8. Instruction set

### 8.1 Command record — `emunpu_command_t` (64 bytes)

| Offset | Size | Field | Description |
|:------:|:----:|-------|-------------|
| 0  | 4  | `opcode`             | `u32` — see §8.2. |
| 4  | 4  | `dtype`              | `u32` — element type (§4). |
| 8  | 16 | `dims[4]`            | `u32[4]` — opcode-specific dimensions (§8.3). |
| 24 | 4  | `first_tile`         | `u32` — index into the tile table (`MATMUL` only). |
| 28 | 4  | `tile_count`         | `u32` — number of tile records (`MATMUL` only). |
| 32 | 12 | `operand_bindings[3]`| `i32[3]` — operand references (§7). |
| 44 | 4  | `result_binding`     | `i32` — result reference (§7). |
| 48 | 12 | `operand_offsets[3]` | `u32[3]` — per-operand byte offsets (§7.1). |
| 60 | 4  | `result_offset`      | `u32` — result byte offset (§7.1). |

### 8.2 Opcode summary

| Mnemonic | Value | Operation |
|----------|:-----:|-----------|
| `MATMUL` | 1 | `result[M,N] = lhs[M,K] × rhs (+= result)` |
| `ADD`    | 2 | `result = lhs + rhs` (elementwise, optional broadcast) |
| `RELU`   | 3 | `result = max(operand, 0)` (elementwise unary) |
| `COPY`   | 4 | `result = operand` (elementwise unary) |

### 8.3 Per-opcode encoding

#### 8.3.1 `MATMUL` (opcode 1)

| Field | Encoding |
|-------|----------|
| `dims` | `{M, N, K, flags}` where `flags` selects the rhs layout (§8.4). |
| `operand_bindings` | `{lhs, rhs, -1}` — `lhs` is `A[M,K]`, `rhs` is `B` (layout per `flags`). |
| `result_binding` | `out` — `C[M,N]`. |
| `first_tile`, `tile_count` | Range into the tile table giving the compile-time schedule (§9). |

Semantics: `C[M,N] = A[M,K] × B`, tiled and distributed per the tile schedule.
`ACCUMULATE` tiles add into the running `C` tile (§9).

#### 8.3.2 `ADD` (opcode 2)

| Field | Encoding |
|-------|----------|
| `dims` | `{elem_count, period, 0, 0}`. |
| `operand_bindings` | `{lhs, rhs, -1}`. |
| `result_binding` | `out`. |
| `first_tile`, `tile_count` | MUST be `0`. |

Semantics: `out[i] = lhs[i] + rhs[i']` for `i` in `[0, elem_count)`. If
`period == 0`, operands are equal-shape (`i' = i`). If `period > 0`, `rhs` is
broadcast as `rhs[i mod period]` — the row-bias pattern used for a `Linear`
layer's bias term.

#### 8.3.3 `RELU` (opcode 3) and `COPY` (opcode 4)

| Field | Encoding |
|-------|----------|
| `dims` | `{elem_count, 0, 0, 0}`. |
| `operand_bindings` | `{in, -1, -1}`. |
| `result_binding` | `out`. |
| `first_tile`, `tile_count` | MUST be `0`. |

Semantics: `RELU` computes `out[i] = max(in[i], 0)`; `COPY` computes
`out[i] = in[i]`, both over `[0, elem_count)`.

### 8.4 MATMUL layout flags (`dims[3]`)

The lhs is always `A[M,K]` and the output is always `C[M,N]`; only the rhs layout
varies.

| Name | Value | rhs layout | Contraction |
|------|:-----:|-----------|-------------|
| `IREE_HAL_EMUNPU_MATMUL_STD`         | 0 | `B[K,N]` | `C[m,n] += A[m,k]·B[k,n]` |
| `IREE_HAL_EMUNPU_MATMUL_TRANSPOSE_B` | 1 | `B[N,K]` | `C[m,n] += A[m,k]·B[n,k]` |

`TRANSPOSE_B` matches the weight layout of `torch.nn.Linear` (`x · Wᵀ`).

## 9. Matmul tile schedule

For each `MATMUL` command, the producer decides the tiling **at compile time**
and emits it as a sequence of tile records. The consumer replays the schedule;
it does not tile at runtime. This mirrors how a GPU backend bakes a workgroup
grid into its executable, and moves the tiling policy to where the target
architecture is known.

A matmul is decomposed into `MMA_DIM × MMA_DIM` output blocks, each accumulated
over the `K` dimension in `MMA_DIM`-wide k-tiles.

### 9.1 Tile record — `emunpu_matmul_tile_t` (32 bytes)

| Offset | Size | Field | Description |
|:------:|:----:|-------|-------------|
| 0  | 4 | `core_id` | Core the tile executes on. MUST be `< IREE_EMUNPU_NUM_CORES`. |
| 4  | 4 | `m0`      | Tile origin along `M` (element offset). |
| 8  | 4 | `n0`      | Tile origin along `N`. |
| 12 | 4 | `k0`      | Tile origin along `K`. |
| 16 | 4 | `tm`      | Tile extent along `M` (`≤ MMA_DIM`; clamped at the matrix edge). |
| 20 | 4 | `tn`      | Tile extent along `N` (`≤ MMA_DIM`). |
| 24 | 4 | `tk`      | Tile extent along `K` (`≤ MMA_DIM`). |
| 28 | 4 | `flags`   | Bitmask of tile flags (§9.2). |

Tile records carry **no absolute memory offsets**: those depend on runtime
buffer staging. The consumer derives per-tile addresses from these
shape-relative coordinates plus the resolved operand base.

### 9.2 Tile flags

| Name | Bit | Meaning |
|------|:---:|---------|
| `IREE_HAL_EMUNPU_TILE_ACCUMULATE` | 0 (value 1) | This k-tile adds into the running `C` block instead of overwriting it. Set on the 2nd..nth k-tile of an output block. |
| `IREE_HAL_EMUNPU_TILE_WRITEBACK`  | 1 (value 2) | Last k-tile of an output block; after it executes, the finished `C` block is written back to global memory. |

The first k-tile of an output block has `ACCUMULATE` clear (it overwrites `C`);
intermediate k-tiles set `ACCUMULATE`; the final k-tile sets both `ACCUMULATE`
(unless it is also the first) and `WRITEBACK`.

## 10. Versioning and compatibility

- The `version` field is a single monotonically increasing integer. There is no
  minor-version negotiation.
- A consumer **MUST** verify `magic` and **MUST** reject a blob whose `version`
  it does not implement.
- Producers and consumers **MUST** be built against the same values of
  `IREE_EMUNPU_MMA_DIM` and `IREE_EMUNPU_NUM_CORES`; these participate in the
  tile-schedule contract and are not encoded in the blob.

## Appendix A — Structure size reference

All sizes assume the little-endian, 4-byte-aligned, padding-free layout of this
format (verified against `reference/emunpu_isa.h`).

| Structure | Size (bytes) |
|-----------|:------------:|
| `emunpu_header_t`      | 28 |
| `emunpu_export_t`      | 24 |
| `emunpu_command_t`     | 64 |
| `emunpu_matmul_tile_t` | 32 |

## Appendix B — Worked example: a fused `Linear + ReLU`

A dispatch computing `y = relu(x · Wᵀ + b)` for `x[M,K]`, `W[N,K]`, `b[N]`,
with bindings `0 = x`, `1 = W`, `2 = b`, `3 = y`, and one scratch slot for the
matmul output:

| # | opcode | dtype | dims | operands | result | tiles |
|:-:|--------|-------|------|----------|--------|-------|
| 0 | `MATMUL` | `F32` | `{M, N, K, TRANSPOSE_B}` | `{0, 1, -1}` | `-2` (scratch 0) | `first_tile..+tile_count` |
| 1 | `ADD`    | `F32` | `{M*N, N, 0, 0}`         | `{-2, 2, -1}` (scratch 0, `b`) | `-2` (scratch 0) | `0` |
| 2 | `RELU`   | `F32` | `{M*N, 0, 0, 0}`        | `{-2, -1, -1}` (scratch 0) | `3` (`y`) | `0` |

Command 1 uses `period = N` so the length-`N` bias `b` is broadcast across all
`M` rows. The tile table entries referenced by command 0 give the
`MMA_DIM × MMA_DIM` block grid and its assignment across the cores.

## Appendix C — (non-normative) proposed v5 opcodes

The device implements a considerably larger operation set than this format can
name (see [`MICRO_ISA.md`](MICRO_ISA.md)). A producer that wants to reach it from
a compiled blob needs a version bump; this appendix records the intended shape of
that change so the two ISAs stay aligned. **None of it is implemented by the v4
loader in [`reference/`](../reference/), which MUST reject a v5 blob** per §10.

Opcode values continue the existing sequence; the coarse ops keep values 1–4.

| Mnemonic | Value | `dims` encoding | Operands → result | Notes |
|----------|:-----:|-----------------|-------------------|-------|
| `ACT`       | 5  | `{elem_count, act_kind, 0, 0}` | `{in, −1, −1}` → `out` | needs two `f32` scalar parameters (alpha/beta), i.e. a new fixed-width `params[2]` field or a constant-table reference |
| `SOFTMAX`   | 6  | `{rows, width, flags, 0}` | `{in, −1, −1}` → `out` | `flags` bit 0 = log-softmax |
| `NORM`      | 7  | `{rows, width, norm_kind, 0}` | `{in, gamma, beta}` → `out` | epsilon as a parameter; `gamma`/`beta` may be `−1` |
| `TRANSPOSE` | 8  | `{rows, cols, 0, 0}` | `{in, −1, −1}` → `out` | |
| `POOL2D`    | 9  | `{batch, channels, pool_kind, 0}` | `{in, −1, −1}` → `out` | window geometry needs a side table analogous to the tile table |
| `CONV2D`    | 10 | `{batch, in_channels, out_channels, act_kind}` | `{in, weight, bias}` → `out` | ditto; the consumer may lower it to im2col + the existing tiled GEMM, or the producer may emit `IM2COL` + `MATMUL` itself |
| `IM2COL`    | 11 | `{batch, patch_width, 0, 0}` | `{in, −1, −1}` → `out` | |
| `GATHER`    | 12 | `{count, width, table_rows, 0}` | `{table, indices, −1}` → `out` | |
| `REDUCE`    | 13 | `{rows, width, reduce_kind, 0}` | `{in, −1, −1}` → `out` | `ARGMAX`/`ARGMIN` need an integer result dtype |

Two format changes are implied and should land together with the opcodes:

1. **Scalar parameters.** `ACT` (alpha/beta) and `NORM` (epsilon) need real
   values, not just dimensions. Adding a `float params[2]` to
   `emunpu_command_t` is the smallest change and keeps every field 4-byte
   aligned.
2. **A window table.** `CONV2D`/`POOL2D`/`IM2COL` carry 9 geometry fields, which
   do not fit in `dims[4]`. The precedent already exists: add a window table
   beside the tile table and have the command hold a `first_window` index, the
   same way `MATMUL` points into the tile schedule.

The dtype enumeration would also widen beyond `F32`/`I32` to cover the seven
element types the hardware supports.
