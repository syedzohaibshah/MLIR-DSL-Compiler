# EmuNPU Hardware (micro-architectural) Instruction Set

**Document status:** Stable
**Normative header:** [`include/emunpu/arch.h`](../include/emunpu/arch.h)
**Implementation:** [`src/core.c`](../src/core.c) (kernels), [`src/ops.c`](../src/ops.c) (sub-op tables + activation transfer function)

---

## Revision history

| Revision | Summary of changes |
|:--------:|--------------------|
| 1 | `DMA_G2L`, `DMA_L2G`, `MMA16`, `MATVEC`, `EW`, `REDUCE`, `FMA`, `CONVERT`. Elementwise kinds add/sub/mul/div/max/min/relu/copy; reductions sum/max/min. |
| 2 | **Activation unit** (`ACT`, 24 kinds with two scalar parameters); **vector-engine ML ops** `SOFTMAX`, `NORM`, `TRANSPOSE`, `IM2COL`, `POOL2D`, `GATHER`; elementwise `POW`; reductions `MEAN`, `PROD`, `ARGMAX`, `ARGMIN`; per-instruction scalar parameters (`param0`/`param1`) and window geometry (`spatial`). |
| 3 | **Per-instruction capability limits for every unit** (§2.3), enforced by `emunpu_instr_validate` and queryable via `emunpu_device_query_caps`. `SOFTMAX`/`NORM`/`REDUCE` now take one row per instruction; `POOL2D` takes one plane's output row band; `TRANSPOSE`/`GATHER`/`IM2COL` are bounded by the matrix tile height; `EW` gained `bcast_phase` so a bounded slice can sit anywhere in a broadcast pattern. Adds the **low-level programming API** (§10). |

---

## 1. Scope

This document specifies the instruction set **the emulated NPU's cores
execute** — the layer below the coarse executable format. There are two ISAs in
this project and they answer different questions:

| | [`ISA.md`](ISA.md) — executable format | **This document** — hardware ISA |
|---|---|---|
| What it is | the `emunpu-cmd-fb` blob a compiler emits | the instruction stream a core executes |
| Granularity | whole-tensor coarse ops (`MATMUL`/`ADD`/`RELU`/`COPY`) | tiled/strided micro-instructions |
| Who produces it | a compiler backend | the runtime's scheduler / device layer |
| Normative header | [`reference/emunpu_isa.h`](../reference/emunpu_isa.h) | [`include/emunpu/arch.h`](../include/emunpu/arch.h) |

The operations added in revision 2 live **in the hardware**: they are reachable
through the device/runtime API ([`include/emunpu/device.h`](../include/emunpu/device.h))
and are what a runtime driver calls. The coarse blob format is deliberately
narrower and is unchanged at version 4; see [§12](#12-reaching-these-ops-from-a-compiled-blob).

Where this prose and `arch.h` disagree, the header governs.

## 2. Execution model

A device has `IREE_EMUNPU_NUM_CORES` (10) independent cores. Each core owns:

- a **local scratchpad** of `IREE_EMUNPU_LOCAL_MEM_BYTES` (256 KB) with a bump
  allocator,
- a private **command queue** of `emunpu_instr_t` records,
- a **16×16 MMA engine** (`IREE_EMUNPU_MMA_DIM`), and
- **DMA** access to the single shared global-memory arena.

The device fills queues and then drains them (`emunpu_core_run`), executing each
instruction in order. Execution is functionally synchronous and bit-exact; there
is no timing model (see [`ARCHITECTURE.md` §6](ARCHITECTURE.md#6-limitations-compared-to-real-hardware)).

### 2.1 Operand addressing

Every operand is a `(space, byte offset)` pair:

| Space | Meaning |
|-------|---------|
| `EMUNPU_SPACE_GLOBAL` | offset into the shared global-memory arena |
| `EMUNPU_SPACE_LOCAL`  | offset into the *issuing core's* scratchpad |

Before touching memory, a kernel resolves each operand through
`emunpu_core_resolve`, which validates that the whole `[offset, offset + bytes)`
window is in range for its space and returns `EMUNPU_ERROR_OUT_OF_RANGE`
otherwise. **Every kernel bounds-checks all of its operands up front**, so a
malformed instruction fails cleanly instead of corrupting memory.

Row strides (`a_row_stride`, `b_row_stride`, `c_row_stride`) are in **bytes**; 0
means "packed" (the natural row width for that operand). Non-zero strides are
what let a core address a sub-block of a larger row-major matrix without copying
it — the mechanism behind tiled MMA reads and the multi-core split of
`TRANSPOSE`.

### 2.2 Element types are orthogonal to the opcodes

All element access goes through `emunpu_dtype_load` / `emunpu_dtype_store`
(see [`src/dtype.c`](../src/dtype.c)), which widen any of the seven dtypes
(`fp32`, `fp16`, `bf16`, `i32`, `i16`, `i8`, `u8`) to `double` and narrow back
with the rounding/saturation appropriate to the target. Consequences that hold
for **every** opcode:

- arithmetic is performed in `double` and rounded once on store;
- mixed precision (e.g. `fp16` inputs → `fp32` output) needs no special case;
- narrowing to an integer dtype rounds to nearest and **saturates** rather than
  wrapping.

An instruction carries four dtypes (`a_dtype`, `b_dtype`, `acc_dtype`,
`c_dtype`); most ops use `a_dtype`/`c_dtype` only, and `acc_dtype` is
informational (the accumulation precision is always `double` in this model).

### 2.3 Per-instruction capability limits

**One instruction is one issue of one functional unit, and a unit can only
process as much data per issue as it is wide.** That bound is part of the ISA:
an instruction asking for more is *rejected* (`emunpu_instr_validate` →
`EMUNPU_ERROR_OUT_OF_RANGE`), never quietly executed. The matrix engine has
always worked this way — a `MMA16` wider than 16 is an error — and every other
unit does too:

| Unit | Opcodes | Limit per instruction | Constant |
|------|---------|----------------------|----------|
| Matrix engine | `MMA16`, `MATVEC` | `M, N, K ≤ 16` | `IREE_EMUNPU_MMA_DIM` |
| Vector ALU | `EW`, `FMA`, `CONVERT` | `M ≤ 256` elements | `IREE_EMUNPU_VECTOR_LANES` |
| Special-function unit | `ACT` | `M ≤ 64` elements | `IREE_EMUNPU_SFU_LANES` |
| Row unit | `REDUCE`, `SOFTMAX`, `NORM` | `M = 1` row, `N ≤ 1024` | `IREE_EMUNPU_MAX_ROW_WIDTH` |
| Tile shuffler | `TRANSPOSE` | `M, N ≤ 16` | `IREE_EMUNPU_MMA_DIM` |
| Spatial | `IM2COL` | `M ≤ 16` patch rows, `N ≤ 1024` | `IREE_EMUNPU_MMA_DIM` |
| Spatial | `POOL2D` | one plane, `M · out_w ≤ 256`, `out_w ≤ 256` | `IREE_EMUNPU_VECTOR_LANES` |
| Gather | `GATHER` | `M ≤ 16` rows, `N ≤ 1024` | `IREE_EMUNPU_MMA_DIM` |
| DMA | `DMA_G2L`, `DMA_L2G` | `M · N · elem ≤ 64 KB` | `IREE_EMUNPU_DMA_MAX_BYTES` |

The SFU is deliberately narrower than the vector ALU (1/4 of it): evaluating a
transcendental costs more silicon per lane than an add, which is why real designs
under-provision it. The row unit takes one row per issue because it has a single
accumulator to stream that row through.

Three consequences shape everything else in this document:

1. **Tiling a tensor-sized operation into unit-sized instructions is the
   producer's job**, not something the hardware hides. `src/device.c` does it for
   the high-level API (§9); a compiler backend or a hand-written kernel does it
   itself (§10).
2. **The limits are discoverable, not assumed.** Read them at runtime from
   `emunpu_device_query_caps()` rather than compiling against the macros, so a
   schedule stays correct on a differently-configured device.
3. **A tile's working set fits a core's scratchpad.** A 16×16 fp32 operand tile is
   1 KB and the scratchpad is 256 KB, so a producer can stage operands through
   local memory (DMA in → compute → DMA out) instead of streaming from global
   memory — see the worked example in §10.

Not everything can be solved by tiling: a `SOFTMAX`/`NORM`/`REDUCE` row wider
than `MAX_ROW_WIDTH` is simply not expressible, because the reduction is not
decomposable into independent slices without a multi-pass algorithm. The device
layer reports `EMUNPU_ERROR_OUT_OF_RANGE` for such a request rather than
pretending; splitting the tensor is the caller's decision.

## 3. Instruction encoding

`emunpu_instr_t` is a flat, zero-initializable record. Fields not used by an
opcode **MUST** be zero.

| Field | Type | Purpose |
|-------|------|---------|
| `op` | `emunpu_iopcode_t` | the opcode ([§4](#4-opcode-reference)) |
| `a_dtype`, `b_dtype`, `acc_dtype`, `c_dtype` | `emunpu_dtype_t` | element types of the two inputs, the accumulator, and the output |
| `a_space`, `b_space`, `c_space` | `emunpu_space_t` | memory space per operand |
| `a_offset`, `b_offset`, `c_offset` | `uint64_t` | byte offset per operand |
| `M`, `N`, `K` | `uint32_t` | problem dimensions, interpreted per opcode |
| `a_row_stride`, `b_row_stride`, `c_row_stride` | `uint32_t` | byte row strides; 0 ⇒ packed |
| `flags` | `uint32_t` | opcode-specific modifier: MMA flag bitmask, or the elementwise / reduction / activation / normalization / pooling **kind**, or the softmax flag bitmask |
| `bcast_period` | `uint32_t` | `EW` only: broadcast period of operand b (0 ⇒ equal shape) |
| `param0`, `param1` | `double` | scalar operands of the sub-operation: `ACT` alpha/beta, `NORM` epsilon |
| `spatial` | `emunpu_spatial_t` | `IM2COL` / `POOL2D` window geometry ([§6](#6-window-geometry-emunpu_spatial_t)) |

### 3.1 Dimension fields by opcode

| Opcode | `M` | `N` | `K` |
|--------|-----|-----|-----|
| `DMA_G2L`, `DMA_L2G` | rows | elements per row | — |
| `MMA16` | rows of C | columns of C | contraction depth |
| `MATVEC` | rows of A | — | length of x |
| `EW`, `FMA`, `CONVERT`, `ACT` | element count | — | — |
| `REDUCE`, `SOFTMAX`, `NORM` | 1 (one row) | row width | — |
| `TRANSPOSE` | rows of A | columns of A | — |
| `IM2COL` | patch rows to emit | patch row width | — |
| `POOL2D` | output rows of one plane | — | — |
| `GATHER` | indices to gather | row width | rows in the table |

Each of these is bounded by §2.3.

## 4. Opcode reference

| Mnemonic | Value | Unit | Operation | Per-issue limit |
|----------|:-----:|------|-----------|-----------------|
| `EMUNPU_I_DMA_G2L`   | 1  | DMA    | strided 2D copy global → local | 64 KB |
| `EMUNPU_I_DMA_L2G`   | 2  | DMA    | strided 2D copy local → global | 64 KB |
| `EMUNPU_I_MMA16`     | 3  | matrix | `C[M×N] (+)= A[M×K] · B` | 16×16×16 |
| `EMUNPU_I_MATVEC`    | 4  | matrix | `y[M] (+)= A[M×K] · x[K]` | 16×16 |
| `EMUNPU_I_EW`        | 5  | vector | elementwise unary/binary, optional broadcast | 256 elems |
| `EMUNPU_I_REDUCE`    | 6  | row    | row reduction to one element | 1 row ≤ 1024 |
| `EMUNPU_I_FMA`       | 7  | vector | `c[i] = a[i]·b[i] + c[i]` | 256 elems |
| `EMUNPU_I_CONVERT`   | 8  | vector | `c[i] = convert(a[i])` | 256 elems |
| `EMUNPU_I_ACT`       | 9  | SFU    | `c[i] = act(a[i])`, parameterized | 64 elems |
| `EMUNPU_I_SOFTMAX`   | 10 | row    | row-wise (log-)softmax | 1 row ≤ 1024 |
| `EMUNPU_I_NORM`      | 11 | row    | row-wise layer / RMS normalization | 1 row ≤ 1024 |
| `EMUNPU_I_TRANSPOSE` | 12 | matrix | `C[N×M] = transpose(A[M×N])` | 16×16 |
| `EMUNPU_I_IM2COL`    | 13 | vector | convolution patch extraction | 16 rows |
| `EMUNPU_I_POOL2D`    | 14 | vector | 2D max / average pooling | 256 out elems |
| `EMUNPU_I_GATHER`    | 15 | vector | `c[i,:] = a[idx[i],:]` | 16 rows |

Opcodes 1–8 are revision 1 and unchanged. The rest are specified below.

### 4.1 `ACT` (9) — activation unit

```
c[i] = act_kind(a[i], param0, param1)     for i in [0, M)
```

| Field | Use |
|-------|-----|
| `M` | element count |
| `flags` | `emunpu_act_kind_t` ([§5](#5-activation-kinds)) |
| `param0`, `param1` | the kind's `alpha` / `beta`; ignored by kinds that take none |
| `a_offset`/`c_offset` | input / output, may alias for an in-place epilogue |

Rejects `flags >= EMUNPU_ACT_COUNT` with `EMUNPU_ERROR_INVALID_ARGUMENT`. The
kind is applied in `double` (see §2.2), so `ACT` on an `fp16` tensor computes at
full precision and rounds once.

### 4.2 `SOFTMAX` (10)

```
row_max = max_j a[r,j]
row_sum = sum_j exp(a[r,j] - row_max)
c[r,j]  = exp(a[r,j] - row_max) / row_sum                     (default)
c[r,j]  = (a[r,j] - row_max) - log(row_sum)                   (EMUNPU_SOFTMAX_LOG)
```

| Field | Use |
|-------|-----|
| `M`, `N` | `M = 1` (one row per issue); `N` = row width, `1 ≤ N ≤ MAX_ROW_WIDTH` |
| `flags` | bitmask; bit 0 = `EMUNPU_SOFTMAX_LOG` |
| `a_row_stride`, `c_row_stride` | optional strides for operating on a sub-matrix |

Subtracting the row maximum is **normative**, not an implementation detail: it is
what makes the op safe for arbitrarily large logits (the test drives a row offset
by +400 to prove `exp()` never overflows).

### 4.3 `NORM` (11)

```
EMUNPU_NORM_LAYER:  mu = mean_j a[r,j]
                    var = mean_j (a[r,j] - mu)^2            (biased / population)
                    c[r,j] = (a[r,j] - mu) / sqrt(var + eps)

EMUNPU_NORM_RMS:    ms = mean_j a[r,j]^2
                    c[r,j] = a[r,j] / sqrt(ms + eps)
```

| Field | Use |
|-------|-----|
| `M`, `N` | `M = 1` (one row per issue); `N` = row width, `1 ≤ N ≤ MAX_ROW_WIDTH` |
| `flags` | `emunpu_norm_kind_t` |
| `param0` | epsilon |

The variance is the **biased** (divide-by-`N`) estimator, matching
`torch.nn.LayerNorm`. The affine `gamma`/`beta` terms are intentionally **not**
part of this instruction: the device layer applies them as period-`N` broadcast
`EW` ops, which keeps the normalization unit single-operand and reuses the
existing broadcast path.

### 4.4 `TRANSPOSE` (12)

```
c[j,i] = a[i,j]     for i in [0, M), j in [0, N)
```

Reads an `M×N` block (stride `a_row_stride`, default `N` elements) and writes the
`N×M` transposed block (stride `c_row_stride`, default `M` elements). Because
both strides are explicit, one instruction can transpose a *row band* of a larger
matrix into the corresponding *column band* of the output — which is exactly how
the device parallelizes a transpose across cores.

`M == 0` or `N == 0` ⇒ `INVALID_ARGUMENT`. The op is a pure permutation: no
arithmetic, so it is exact even across a dtype change.

### 4.5 `IM2COL` (13)

Turns a CHW image into the patch matrix that makes a convolution a GEMM.

```
patch  = spatial.first_out_row + r          (r in [0, M))
oy     = patch / spatial.out_w
ox     = patch % spatial.out_w
col    = (ch * kernel_h + ky) * kernel_w + kx
iy     = oy * stride_h - pad_h + ky
ix     = ox * stride_w - pad_w + kx
c[r,col] = in_bounds(iy, ix) ? a[(ch * in_h + iy) * in_w + ix] : 0
```

| Field | Use |
|-------|-----|
| `M` | patch rows this instruction emits |
| `N` | patch row width; **MUST** equal `channels * kernel_h * kernel_w` |
| `spatial` | window geometry, including `first_out_row` |
| `c_row_stride` | optional stride of the patch matrix |

`first_out_row` is what allows a row range to be split across cores: a kernel
needs the *logical* patch index (not just a write offset) to recover the window
position. Errors: a mismatched `N`, a zero `channels`/`kernel_*`/`out_*`, or
`first_out_row + M > out_h * out_w` (⇒ `OUT_OF_RANGE`).

The column order `(channel, kernel_y, kernel_x)` is normative — it matches the
memory layout of a `[out_channels, channels, kernel_h, kernel_w]` convolution
weight, so the GEMM can use such a weight **as-is** with `TRANSPOSE_B`.

### 4.6 `POOL2D` (14)

```
oy            = spatial.first_out_row + r          (r in [0, M))
window(oy,ox) = { a[iy, ix] : iy = oy*stride_h - pad_h + ky, in bounds
                              ix = ox*stride_w - pad_w + kx, in bounds }
EMUNPU_POOL_MAX: c[r,ox] = max(window)
EMUNPU_POOL_AVG: c[r,ox] = sum(window) / count(window)
```

One instruction covers a **row band of one plane**: it emits `M` output rows
starting at `spatial.first_out_row`, each `out_w` wide. `a_offset` addresses the
input plane; `c_offset` addresses the first output row written.

| Field | Use |
|-------|-----|
| `M` | output rows to emit (`M · out_w ≤ VECTOR_LANES`) |
| `flags` | `emunpu_pool_kind_t` |
| `spatial` | window geometry + `first_out_row` (`channels` unused: the plane is selected by the operand offsets) |

Padding taps are **excluded**, not treated as zero: `MAX` never sees them, and
`AVG` divides by the number of in-bounds taps (PyTorch's
`count_include_pad=False`). An entirely out-of-bounds window yields 0.

### 4.7 `GATHER` (15)

```
c[i,:] = a[idx[i],:]     for i in [0, M), idx = operand b
```

| Field | Use |
|-------|-----|
| `M`, `N`, `K` | index count, row width, **table row count** |
| `b_dtype` | index element type (normally `i32`) |
| `a_row_stride`, `c_row_stride` | optional strides |

Every index is checked against `[0, K)`; a violation returns
`EMUNPU_ERROR_OUT_OF_RANGE` before any out-of-bounds read. `M == 0` is a no-op.

## 5. Activation kinds

`flags` selects one of `emunpu_act_kind_t`; `param0` is `alpha` and `param1` is
`beta`. The transfer function is `emunpu_act_apply` in
[`src/ops.c`](../src/ops.c), which is exported so a host-side reference model can
reproduce the device exactly.

| Kind | Value | Formula | Params |
|------|:-----:|---------|--------|
| `RELU`        | 0  | `max(x, 0)` | — |
| `LEAKY_RELU`  | 1  | `x > 0 ? x : alpha·x` | `alpha` |
| `RELU6`       | 2  | `clamp(x, 0, 6)` | — |
| `CLAMP`       | 3  | `clamp(x, alpha, beta)` | `alpha`, `beta` |
| `SIGMOID`     | 4  | `1 / (1 + e^-x)` | — |
| `TANH`        | 5  | `tanh(x)` | — |
| `GELU`        | 6  | `x·½·(1 + erf(x/√2))` (exact) | — |
| `GELU_TANH`   | 7  | `½x·(1 + tanh(√(2/π)·(x + 0.044715x³)))` | — |
| `SILU`        | 8  | `x·sigmoid(x)` (swish) | — |
| `ELU`         | 9  | `x > 0 ? x : alpha·(e^x − 1)` | `alpha` |
| `SOFTPLUS`    | 10 | `log(1 + e^x)` | — |
| `HARDSIGMOID` | 11 | `clamp(x/6 + ½, 0, 1)` | — |
| `HARDSWISH`   | 12 | `x·clamp(x/6 + ½, 0, 1)` | — |
| `MISH`        | 13 | `x·tanh(softplus(x))` | — |
| `EXP`         | 14 | `e^x` | — |
| `LOG`         | 15 | `ln x` | — |
| `SQRT`        | 16 | `√x` | — |
| `RSQRT`       | 17 | `1/√x` | — |
| `ABS`         | 18 | `abs(x)` | — |
| `NEG`         | 19 | `−x` | — |
| `RECIP`       | 20 | `1/x` | — |
| `SIGN`        | 21 | `−1 / 0 / +1` | — |
| `SQUARE`      | 22 | `x²` | — |
| `ERF`         | 23 | `erf(x)` | — |

Notes on numerics, all of which the unit implements deliberately:

- `SIGMOID` uses the two-sided form (`e^x/(1+e^x)` for `x < 0`) so a large
  negative input cannot overflow `exp`.
- `SOFTPLUS` (and therefore `MISH`) saturates to its asymptotes outside
  `|x| > 30` instead of overflowing.
- `LOG`, `SQRT`, `RSQRT`, `RECIP` have restricted domains and propagate
  IEEE `NaN`/`inf` rather than reporting an error — same as the hardware they
  model.
- Every formula follows PyTorch's definition, so a model exported from PyTorch
  maps 1:1.

## 6. Window geometry (`emunpu_spatial_t`)

| Field | Meaning |
|-------|---------|
| `in_h`, `in_w` | input plane extents |
| `channels` | channels per image (`IM2COL`; unused by `POOL2D`) |
| `kernel_h`, `kernel_w` | window extents |
| `stride_h`, `stride_w` | window step |
| `pad_h`, `pad_w` | zero padding on each side of each axis |
| `out_h`, `out_w` | **derived** output extents (see below) |
| `first_out_row` | `IM2COL` only: logical index of the first patch row emitted |

The output extent of an axis is
`floor((in + 2·pad − kernel) / stride) + 1`, or 0 when the window does not fit;
`emunpu_spatial_out_extent()` in `arch.h` is the single normative
implementation. The producer computes `out_h`/`out_w` **once** and stores them in
the instruction so every core agrees on the same grid.

Planes are CHW-contiguous: channel `c` of an image starts at `c · in_h · in_w`
elements.

## 7. Other sub-operation kinds

### 7.1 Elementwise (`emunpu_ew_kind_t`)

| Kind | Value | Operation | Arity |
|------|:-----:|-----------|-------|
| `ADD`  | 0 | `a + b` | binary |
| `SUB`  | 1 | `a − b` | binary |
| `MUL`  | 2 | `a · b` | binary |
| `DIV`  | 3 | `a / b` | binary |
| `MAX`  | 4 | `max(a, b)` | binary |
| `MIN`  | 5 | `min(a, b)` | binary |
| `RELU` | 6 | `max(a, 0)` | unary |
| `COPY` | 7 | `a` | unary |
| `POW`  | 8 | `pow(a, b)` | binary |

Arity is **not** a value range — query `emunpu_ew_is_binary(kind)`. (`RELU` and
`COPY` sit between the binary kinds for backward compatibility; new binary kinds
are appended after them.)

`bcast_period` `p > 0` indexes operand b as `b[(bcast_phase + i) mod p]`, where
`bcast_phase` is the global element index the instruction's slice starts at.
`p = N` is the row-bias / per-channel pattern; `p = 1` broadcasts a scalar. Unary
kinds ignore operand b entirely.

The phase exists because of §2.3: an elementwise op over a whole tensor is split
into `VECTOR_LANES`-sized instructions, and such a slice generally does *not*
start on a period boundary — it cannot, if the period is wider than the vector
unit. Carrying the phase makes any slice of any broadcast pattern issuable.

### 7.2 Reductions (`emunpu_reduce_kind_t`)

| Kind | Value | Result per row |
|------|:-----:|----------------|
| `SUM`    | 0 | `Σ a[r,j]` |
| `MAX`    | 1 | `max a[r,j]` |
| `MIN`    | 2 | `min a[r,j]` |
| `MEAN`   | 3 | `Σ a[r,j] / N` |
| `PROD`   | 4 | `Π a[r,j]` |
| `ARGMAX` | 5 | index of the maximum (first occurrence) |
| `ARGMIN` | 6 | index of the minimum (first occurrence) |

`ARGMAX`/`ARGMIN` emit a **position**, so the output tensor is normally an
integer dtype; ties resolve to the lowest index, as in PyTorch.

### 7.3 Normalization and pooling kinds

`emunpu_norm_kind_t`: `LAYER` (0), `RMS` (1).
`emunpu_pool_kind_t`: `MAX` (0), `AVG` (1).
`SOFTMAX` flags: `EMUNPU_SOFTMAX_LOG` (bit 0).

## 8. How the device API lowers onto these instructions

[`device.h`](../include/emunpu/device.h) is the runtime-facing API. It is a
*convenience layer over this ISA*: each entry point tiles its tensor-sized
operation into instructions no larger than §2.3 allows, then deals them across the
cores of the current execution scope (§9). Composite ops decompose at
*submission* time, so a stream records the whole expansion.

| Device call | Micro-instructions issued (one per tile) |
|-------------|---------------------------|
| `emunpu_activation` | `ACT` × ⌈count / SFU_LANES⌉ |
| `emunpu_softmax` | `SOFTMAX` × rows |
| `emunpu_layernorm` | `NORM(LAYER)` × rows → `EW(MUL, period=width)` → `EW(ADD, period=width)` |
| `emunpu_rmsnorm` | `NORM(RMS)` × rows → `EW(MUL, period=width)` |
| `emunpu_transpose` | `TRANSPOSE` × 16×16 blocks |
| `emunpu_pool2d` | `POOL2D` × (planes × output row bands) |
| `emunpu_im2col` | `IM2COL` × (images × ⌈patches / 16⌉) |
| `emunpu_conv2d` | `IM2COL` → (`DMA`+`MMA16`)* → `EW(ADD, period=out_channels)` → `ACT` → `TRANSPOSE` per image |
| `emunpu_gather` | `GATHER` × ⌈count / 16⌉ |
| `emunpu_linear` | (`DMA`+`MMA16`)* → `EW(ADD, period=N)` → `ACT` |
| `emunpu_matmul` / `emunpu_gemm` | `DMA_G2L`·2 → `MMA16` per K-tile → `DMA_L2G` |

`conv2d` is the interesting one: it allocates a patch matrix and a GEMM result in
global memory, lowers the convolution to `im2col` + a `TRANSPOSE_B` GEMM **on the
same 16×16 tensor cores** as any other matmul, applies bias and activation as
epilogues, transposes each image's `[patches, out_channels]` result back to NCHW,
and then releases the scratch — immediately on the default stream, or via a
queued release command when the work was deferred to a stream.

## 9. Multi-core partitioning

Tiling to the unit limits (§2.3) produces a list of instructions; those
instructions are then dealt **round-robin across the cores of the execution
scope**. The tiling axis is chosen so each instruction is self-contained:

| Op | Tiling axis | Why it is safe |
|----|-------------|----------------|
| `MMA16` (matmul) | output 16×16 tiles | tiles are independent; the K-tiles of one output tile accumulate in order **on a single core** |
| `MATVEC` | 16-row blocks × 16-wide K-tiles | K-tiles of a row block stay on one core (they accumulate into the same `y`) |
| `EW`, `FMA`, `CONVERT` | 256-element ranges | elementwise |
| `EW` with `bcast_period` | 256-element ranges + `bcast_phase` | the phase makes the broadcast index computable from any starting offset |
| `ACT` | 64-element ranges | elementwise, on the narrower SFU |
| `REDUCE`, `SOFTMAX`, `NORM` | one row per instruction | a row's max/mean/variance must be complete in one issue |
| `TRANSPOSE` | 16×16 blocks | disjoint outputs, addressed via `c_row_stride` |
| `IM2COL` | ≤16 patch rows, per image | rows are independent; `first_out_row` carries the position |
| `POOL2D` | output row bands of one plane | bands are independent; `first_out_row` carries the position |
| `GATHER` | ≤16 indices | rows are independent |

Ragged edges are normal: the final tile of an axis is simply shorter. The tests
use sizes that are deliberately *not* multiples of either the unit widths or the
core count (137 elements, 23 rows, 37×19 transposes, K=27 convolutions) and pass
unchanged for `IREE_EMUNPU_NUM_CORES` of 1, 2, 3, 7, or 10.

**Choosing the cores.** By default the scope is every online core. Narrow it with
`emunpu_device_set_exec_scope(device, first_core, core_count)` to dedicate part of
the device to an operation, partition the device between pipelines, or measure
scaling; `emunpu_device_core_stats()` reports what each core actually did. The
scope is captured when a command is *submitted*, so streamed work replays the
scope that was in force when it was recorded. `demo/lowlevel_demo.c` runs one GEMM
under scopes of 1, 2, 4 and 8 cores and prints the resulting per-core instruction
counts.

## 10. Programming the cores directly

The device API tiles and distributes for you. The low-level API is the same
hardware with that layer removed — you choose the tiling, the core assignment, and
whether operands live in global memory or are staged through each core's
scratchpad. This is the surface a code generator targets.

| Call | Purpose |
|------|---------|
| `emunpu_device_query_caps` | the limits and sizes to plan against (§2.3) |
| `emunpu_global_mem_offset` | a tensor's address in the global address space |
| `emunpu_device_local_alloc` / `_local_reset` | carve up a core's scratchpad |
| `emunpu_device_enqueue` | append one validated instruction to a core's queue |
| `emunpu_device_launch` | execute every core's queue, in order per core |
| `emunpu_device_core_stats` | per-core instructions / MMA issues / DMA bytes |
| `emunpu_instr_validate` | check an instruction without running it |

`emunpu_device_enqueue` validates against §2.3 at *enqueue* time, so an
over-large instruction is reported where it can still be fixed rather than
part-way through a launch.

### 10.1 Worked example: a hand-scheduled GEMM tile

The full kernel is `demo/lowlevel_demo.c` (and `test_raw_programming` in
`test/ops_test.c`, which checks it against a host reference). For one output tile
on one core, staging both operands through the scratchpad:

```c
emunpu_caps_t caps;
emunpu_device_query_caps(dev, &caps);
const uint32_t tile = caps.mma_dim;             // 16

uint64_t a_base, b_base, c_base;                 // global addresses
emunpu_global_mem_offset(gm, dA, &a_base);
emunpu_global_mem_offset(gm, dB, &b_base);
emunpu_global_mem_offset(gm, dC, &c_base);

uint64_t a_loc, b_loc, c_loc;                    // scratchpad tiles
emunpu_device_local_alloc(dev, core, tile * tile * sizeof(float), &a_loc);
emunpu_device_local_alloc(dev, core, tile * tile * sizeof(float), &b_loc);
emunpu_device_local_alloc(dev, core, tile * tile * sizeof(float), &c_loc);

for (uint32_t k0 = 0; k0 < K; k0 += tile) {
  uint32_t tk = min(tile, K - k0);
  // A[m0:+tm, k0:+tk] : strided global read -> packed local tile.
  emunpu_instr_t dma = {0};
  dma.op = EMUNPU_I_DMA_G2L;
  dma.a_dtype = dma.c_dtype = EMUNPU_DT_FP32;
  dma.M = tm; dma.N = tk;
  dma.a_offset = a_base + (m0 * K + k0) * sizeof(float);
  dma.a_row_stride = K * sizeof(float);          // source is a sub-block
  dma.c_offset = a_loc;                          // destination is packed
  emunpu_device_enqueue(dev, core, &dma);
  ... same for B into b_loc ...

  // C_tile (+)= A_tile * B_tile, entirely inside the scratchpad.
  emunpu_instr_t mma = {0};
  mma.op = EMUNPU_I_MMA16;
  mma.a_dtype = mma.b_dtype = mma.acc_dtype = mma.c_dtype = EMUNPU_DT_FP32;
  mma.M = tm; mma.N = tn; mma.K = tk;
  mma.a_space = mma.b_space = mma.c_space = EMUNPU_SPACE_LOCAL;
  mma.a_offset = a_loc; mma.b_offset = b_loc; mma.c_offset = c_loc;
  mma.flags = k0 ? EMUNPU_MMA_ACCUMULATE : 0;    // first k-tile overwrites
  emunpu_device_enqueue(dev, core, &mma);
}
// The finished tile : local -> strided global write.
emunpu_instr_t out = {0};
out.op = EMUNPU_I_DMA_L2G;
out.a_dtype = out.c_dtype = EMUNPU_DT_FP32;
out.M = tm; out.N = tn;
out.a_offset = c_loc;
out.c_offset = c_base + (m0 * N + n0) * sizeof(float);
out.c_row_stride = N * sizeof(float);
emunpu_device_enqueue(dev, core, &out);

emunpu_device_launch(dev);                       // now it all runs
```

Two properties worth noting:

- **Instruction order within a core is the only ordering guarantee.** A core
  executes its queue in order, which is what makes the `ACCUMULATE` chain above
  correct. Nothing orders one core against another, so anything a core
  accumulates into must be produced entirely by that core.
- **Scratchpad reuse is legal and expected.** The three tile buffers above are
  allocated once per core and reused by every output tile that core handles,
  because the DMA that overwrites `a_loc` cannot run until the MMA that read it
  has finished.

## 11. Error reporting

Kernels return `emunpu_status_t`. The first failure aborts the rest of that
core's queue, and the device drops (rather than executes) the queues of the cores
that had not run yet — a broken op must not leave work behind to fire during the
next one. The device is therefore usable again after a failed op.

| Status | Raised when |
|--------|-------------|
| `INVALID_ARGUMENT` | unknown sub-op kind, zero-width row op, `IM2COL` width ≠ `C·kh·kw`, degenerate window, invalid dtype |
| `OUT_OF_RANGE` | **an instruction exceeds its unit's per-issue capability (§2.3)**, an operand window escapes its memory space, a gather index ≥ table rows, `IM2COL`/`POOL2D` rows past the output grid, a row op wider than `MAX_ROW_WIDTH` |
| `UNIMPLEMENTED` | unknown opcode |

The capability check runs first, before any operand is touched — see
`emunpu_core_execute`. With `IREE_EMUNPU_TRACE=1` a rejected instruction is
logged as `REJECTED <op>: exceeds unit capability`.

## 12. Reaching these ops from a compiled blob

The coarse executable format ([`ISA.md`](ISA.md)) is still **version 4** and
still exposes only `MATMUL`, `ADD`, `RELU`, `COPY`. A compiler that emits that
blob therefore cannot yet name an activation kind or a convolution, even though
the hardware implements them — those ops are reached through the device API
(which is what a runtime driver calls directly, and what every test and demo
here uses).

Extending the blob is deliberately a separate step, because it changes a contract
with a *producer* that lives outside this repository. `ISA.md`
[Appendix C](ISA.md#appendix-c--non-normative-proposed-v5-opcodes) sketches the
version-5 opcode assignments and the fields each would need.

## 13. Adding an opcode

The checklist the revision-2 and -3 ops followed, in order:

1. **`include/emunpu/arch.h`** — add the `emunpu_iopcode_t` value (append; never
   renumber), any sub-op enum + its `_COUNT` sentinel, and document which
   `M`/`N`/`K`/stride/`param`/`spatial` fields it reads. Decide **which unit
   issues it and how much it can do per issue** (§2.3); add a constant if the
   unit is new.
2. **`src/ops.c`** — add the mnemonic to `emunpu_iop_name`, a `case` to
   `emunpu_instr_validate` enforcing the per-issue limit, and, if it has kinds, a
   name function for them. Names are shared by the trace, the visualizer, and the
   tests, so there is exactly one table per enum. Surface any new limit in
   `emunpu_caps_t` so producers can discover it.
3. **`src/core.c`** — write the kernel: resolve every operand with
   `emunpu_core_resolve` *before* the first access, read/write through
   `emunpu_dtype_load/store`, then add the `case` to `emunpu_core_execute` and a
   line to `emunpu_core_trace`.
4. **`src/viz.c`** — if it has a sub-operation, add it to `emunpu_viz_subname`,
   and bucket the opcode in `demo/visualizer.py`'s `op_class`.
5. **`include/emunpu/device.h` + `src/device.c`** — add the runtime entry point,
   an `emunpu_cmd_kind_t`, and an `emunpu_exec_*` that fans the work across the
   cores along a safe axis ([§9](#9-multi-core-partitioning)).
6. **`test/ops_test.c`** — check it against an *independently written* host
   reference, at a size that is neither a multiple of the unit width nor of the
   core count, plus the error cases: add it to `test_instruction_limits` so the
   largest legal instruction is accepted and one element more is rejected.
7. **This document** — the opcode table, the field encoding, and the semantics.
