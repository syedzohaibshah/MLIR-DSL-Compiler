# Bytecode contract: compiler → runtime

**Version:** 0.3 (draft for agreement)
**Date:** 2026-11-18
**Owner (runtime):** Syed Zohaib Shah
**For:** the compiler team (MLIR lowering and bytecode emission)

This document specifies the bytecode that the compiler must emit, so that the runtime can run it on emunpu-sim. It also states what the runtime guarantees in return. When both teams agree, this document becomes version 1.0. After that, a change to it needs agreement from both teams.

Each item has a status:

- **Implemented**: the runtime accepts it today.
- **Planned**: agreed design; the runtime does not accept it yet.
- **Open**: needs a decision from both teams. See [Open items](#9-open-items).

Runtime details are in [`bytecode-vm-project.md`](bytecode-vm-project.md) and [`../bytecode-vm/validation-design.md`](../bytecode-vm/validation-design.md).

## Contents

1. [File format](#1-file-format)
2. [Header: inputs and outputs](#2-header-inputs-and-outputs)
3. [Operands](#3-operands)
4. [Instructions](#4-instructions)
5. [LAUNCH on emunpu](#5-launch-on-emunpu)
6. [Rules the compiler must follow](#6-rules-the-compiler-must-follow)
7. [Limits](#7-limits)
8. [What the runtime guarantees](#8-what-the-runtime-guarantees)
9. [Open items](#9-open-items)
10. [Example](#10-example)

## 1. File format

**Status: implemented.**

- Plain text. One instruction on each line: an opcode name, then operands separated by spaces or tabs.
- An empty line is ignored.
- `//` starts a comment. The comment continues to the end of the line. It can be on its own line or after an instruction.
- Every operand is a decimal integer that fits in 32 bits (signed).
- Instructions are numbered from **0**, in file order. Header lines, empty lines, and comment lines do not get a number. A jump target is this 0-based number.
- The last instruction must be `EOF`.

## 2. Header: version, inputs and outputs

**Status: implemented** in the runtime (2026-11-11). The compiler team must confirm it.

The file starts with a version line. Then come one line for each input slot and one line for each output slot. Header lines come before the first instruction.

```
.version 1
.input  <slot> <bytes> <name>
.output <slot> <bytes> <name>
```

| Field | Meaning |
|---|---|
| `.version 1` | The format version. It is the first line. The runtime rejects any other version. |
| `slot` | Slot number. The host uses the same number to give an input or read an output. |
| `bytes` | The exact size of the data in that slot, in bytes. |
| `name` | The name of the tensor in that slot, for example `layers.0.wq`. The host uses it to find the weight. The runtime does not use it for checks. |

A name has only letters, digits, `_`, `.` and `-`. It has no spaces.

**Why the version line.** The version tells the runtime which format the file uses. A later format may add fields, such as dtype and shape. An old runtime that reads a new file would misread those fields. The runtime checks the version first, so it stops with a clear error before it reads anything else.

**Why the name.** The slot and the size tell the runtime what to check. The name tells the host what to load. Without it, the host must hard-code which weight goes in which slot. Then any change in slot numbering on the compiler side breaks the host without any error. The name removes that coupling. The runtime does not check names. It stores them and uses them in messages.

Rules:

- Every program has a header. A file without `.version 1` is an error.
- Each slot number appears at most once for `.input` and at most once for `.output`. Slot numbers do not need to be in order, and they do not need to be contiguous.
- Each declared input must be read by at least one `COPY_H2D`. Each declared output must be written by at least one `COPY_D2H`. An unused declaration is an error.
- Each `COPY_H2D` must read a declared input slot. Each `COPY_D2H` must write a declared output slot.
- The host must set every declared input before the run starts. A missing input is an error at run start.
- The runtime rejects a host input whose size is not equal to the declared size.

A later version adds dtype and shape: `.input <slot> <dtype> <shape> <name>`, for example `.input 0 f32 4x4 x`. See [Open items](#9-open-items).

### Why the header exists

The header declares every slot before the first instruction runs. The runtime checks the host and the program at load time. It finds a missing input, an unused slot, or a wrong size before any device work starts. Without the header, these errors appear only during the run.

## 3. Operands

**Status: implemented.**

An operand is written in one of two forms:

| Form | Example | Meaning |
|---|---|---|
| Number | `16` | The value 16. |
| Register | `@3` | The value that is in register 3 when the instruction runs. |

Each operand has a kind:

| Kind | Meaning | Forms allowed |
|---|---|---|
| **value** | The data that the instruction uses: a size, a buffer id, a slot, a number to store. | number or `@N` |
| **id** | A register id, or a jump target (an instruction number). | number only |

`@N` on an id operand is an error.

## 4. Instructions

**Status: implemented.**

| Instruction | Operands (kind) | Effect |
|---|---|---|
| `ALLOC size` | value | Allocates a device buffer of `size` bytes. Buffer ids are given in order: the first `ALLOC` gives 0, the next gives 1, and so on. |
| `FREE buf` | value | Releases buffer `buf`. |
| `COPY_H2D buf slot size` | value, value, value | Copies `size` bytes from input slot `slot` into buffer `buf`. |
| `COPY_D2H slot buf size` | value, value, value | Copies `size` bytes from buffer `buf` into output slot `slot`. The slot comes first. |
| `LAUNCH op_id n_buf buf… n_scalar scalar…` | all value | Runs a kernel. See [LAUNCH on emunpu](#5-launch-on-emunpu). |
| `SET reg value` | id, value | Writes `value` into register `reg`. |
| `INC reg` | id | Adds 1 to register `reg`. |
| `DEC reg` | id | Subtracts 1 from register `reg`. |
| `BR target` | id | Jumps to instruction `target`. |
| `COND_BR reg if_zero if_nonzero` | id, id, id | If register `reg` holds 0, jumps to `if_zero`. If not, jumps to `if_nonzero`. |
| `EOF` | — | Ends the program with success. |

A register must be written with `SET` before it is read.

## 5. LAUNCH on emunpu

**Status: implemented.**

Format:

```
LAUNCH <op_id> <n_buf> <buf_0> … <buf_n-1> <n_scalar> <scalar_0> … <scalar_m-1>
```

All buffers hold fp32 data. Buffers are flat (rank 1); a matrix is stored row-major.

| op_id | Name | Buffers | Scalars | Computes |
|---|---|---|---|---|
| 0 | MATMUL | `A, B, C` | `M, N, K` | `C[M×N] = A[M×K] · B[K×N]` |
| 1 | VEC_ADD | `A, B, C` | `count` | `C[i] = A[i] + B[i]` for `count` elements |
| 2 | RELU | `A, C` | `count` | `C[i] = max(0, A[i])` for `count` elements. Forwards to the same `emunpu_activation` call ACTIVATION (op id 7) uses, with `kind` fixed to RELU — not a separate implementation. |
| 3 | BIAS_ADD | `A, bias, C` | `count, out_features` | `C[i] = A[i] + bias[i mod out_features]` for `count` elements |
| 4 | RMS_NORM | `A, C` **or** `A, gamma, C` | `rows, width` | `C[r][i] = A[r][i] / sqrt(mean(A[r]^2) + eps) * gamma[i]` for each of `rows` rows of `width` elements. `eps` is fixed at `1e-5`, not an operand. |
| 5 | SOFTMAX | `A, C` | `rows, width, log_softmax` | Row-wise softmax over `rows` rows of `width` elements. `log_softmax = 0` gives `softmax(A)`; `log_softmax = 1` gives `log(softmax(A))`. |
| 6 | TRANSPOSE | `A, C` | `rows, cols` | `C[cols×rows] = transpose(A[rows×cols])`. `rows` and `cols` describe the **input**. Both buffers hold `rows·cols` elements; only the shape differs. |
| 7 | ACTIVATION | `A, C` | `count, kind, alpha_bits, beta_bits` | `C[i] = act(A[i], alpha, beta)` for `count` elements. `kind` selects one of emunpu's 24 activation kinds (0 = RELU, 8 = SiLU, …; see `emunpu/arch.h`'s `emunpu_act_kind_t`). `alpha_bits`/`beta_bits` are `FLOAT_BITS` scalars — always present, even for kinds that ignore them (pass the bits of `0.0f`). Only `LEAKY_RELU`/`ELU` (alpha only) and `CLAMP` (both) read them. See the `FLOAT_BITS` row below. |

RMS_NORM's `gamma` (the learned per-column scale) is optional. `n_buf = 2` (`A, C`) skips it; `n_buf = 3` (`A, gamma, C`) applies it. The runtime picks the meaning of each buffer slot from `n_buf`, the same way ACTIVATION (`n_buf = 2`) and VEC_ADD (`n_buf = 3`) already differ. No new LAUNCH encoding was needed for this.

**O9 decided (2026-11-21): one shared op id per emunpu call family, with the kind as an `ENUM` scalar.** ACTIVATION (op id 7, above) is the first instance — it covers RELU, SiLU, GELU, sigmoid, tanh and 15 more kinds behind one op id, instead of one op id per kind. op id 2 (RELU) keeps its own LAUNCH shape (`count` only, no `kind` scalar) for compatibility, but no longer has its own implementation — `dispatch()` forwards it into the same code path as ACTIVATION, with `kind` fixed rather than read from `args`. See O9's entry below. The same shape will be used for element-wise ops (`emunpu_vector_op`'s 9 kinds) and reductions (`emunpu_reduce`'s 5 kinds) when they are wired.

The remaining Llama-3 operators (element-wise multiply, RoPE, gather) are not available yet. Their op ids and operands will be added to this table by agreement.

### 5.1 Scalar kinds

Not every `LAUNCH` scalar is a dimension. The runtime keeps one **signature row per op_id**
(`KERNEL_TABLE` in `hal/emunpu_device.cpp`) that names each scalar position's kind, and checks
the range that kind allows. A blanket rule over all scalars is wrong: SOFTMAX's `log_softmax`
is legitimately `0`, while MATMUL's `M` never is.

| Kind | Allowed values | Used by today | Why |
|---|---|---|---|
| `DIM` | `> 0` | every scalar of op ids 0–4, 6; `rows`, `width` of op id 5; `count` of op id 7 | A count or a dimension. A zero-size tensor cannot be allocated (R1), so a zero is a lowering bug. |
| `FLAG` | `0` or `1` | `log_softmax` (op id 5) | A mode selector. Both values are meaningful. |
| `PARAM` | any `int`, negative allowed | none yet | A tuning value, for example a signed offset. Reserved so a future kernel does not need a format change. |
| `ENUM` | `0` to `count-1`, where `count` is declared per scalar position | `kind` (op id 7) | Selects one of a fixed, kernel-specific set of kinds. `DIM` would wrongly reject kind `0` (e.g. `ACT_RELU = 0`), the same mistake the old blanket zero-rule made with `log_softmax`. The valid count travels with the scalar position (`ScalarSpec::enum_count` in `hal/emunpu_device.h`), since it differs per kernel — 24 for ACTIVATION (`EMUNPU_ACT_COUNT` in `emunpu/arch.h`), 9 for the planned element-wise op, 5 for the planned reduce op. |
| `FLOAT_BITS` | any 32-bit pattern | `alpha_bits, beta_bits` (op id 7) | Carries a `float`'s raw bits inside an `int` scalar — not a numeric value. The emitter does `memcpy(&bits, &my_float, 4)`; the device does `memcpy(&my_float, &bits, 4)` to recover the exact value, then widens to `double` before calling `emunpu_activation`. A C-style cast (`(int)my_float`) would be wrong here — it converts the *value* (truncates), destroying the float; `memcpy` only relocates bytes. `0` is the correct literal for "pass `0.0f`" (all-zero bytes decode to `0.0f` exactly), which is why kinds that ignore alpha/beta still pass literal `0`s for those scalars. |

**Adding a kernel.** Add one row to `KERNEL_TABLE` naming the buffer-count range and the kind of
each scalar, then write its `do_*` body. The generic check in `dispatch` needs no change, and the
`do_*` body does not repeat the count checks — the table has already guaranteed them.

**Non-integer tuning values.** Every `LAUNCH` argument is an `int`, so a float constant such as
RMS_NORM's `eps` cannot travel in the bytecode today; it is fixed inside the kernel. If a kernel
ever needs a float the compiler chooses, the planned encoding is a `FLOAT_BITS` scalar kind: the
compiler emits the float's 32-bit pattern as an `int`, and the device reads it back. The format
stays integer-only. This is not implemented; see `PROGRESS.md`.

**Where each `LAUNCH` rule is checked.** The split follows D7: a device-independent fact is
checked in the verifier, a kernel-specific fact in the device.

| Checked | Where | Why there |
|---|---|---|
| Arity ≥ 2; every argument is a VALUE | verifier | True of `LAUNCH` itself, whatever the kernel. |
| `op_id`, `n_buf`, every buffer id, `n_scalar` are not negative | verifier | Structural. No kernel can make a negative count or id meaningful. |
| Total equals `2 + n_buf + 1 + n_scalar` | device | Needs `n_buf` and `n_scalar` resolved, including `@N`. |
| Buffer count is in the kernel's range | device | Needs the signature row. |
| Each scalar's range, by its kind | device | Needs the signature row: which position is a `DIM`, a `FLAG` or a `PARAM`. |

The verifier deliberately does **not** check the trailing scalars. Their valid range depends on
the kernel, and the signature table that knows this is emunpu-specific, while the verifier is
device-independent. A new op_id therefore needs no verifier change — only a table row.

## 6. Rules the compiler must follow

| # | Rule | Status | Reason |
|---|---|---|---|
| R1 | An `ALLOC` size is greater than 0 and is a multiple of 4. | implemented | emunpu stores every buffer as fp32. |
| R2 | A copy size **equals** the declared slot size. A partial copy is an error. | implemented | Keeps the binding simple. Partial copies come later, with copy offsets for tiling. |
| R3 | A copy size is not larger than the buffer size. | implemented | Stops a copy past the end of a buffer. |
| R4 | A LAUNCH reads and writes only inside its buffers. For example, for MATMUL: `M·K·4 ≤ size(A)`, `K·N·4 ≤ size(B)`, `M·N·4 ≤ size(C)`. | implemented | The device does not stop a kernel that writes past a buffer. The runtime checks every buffer before each launch. |
| R5 | Do not use a buffer after `FREE`, and do not free it twice. | implemented | The runtime rejects it. |
| R6 | Free every buffer before `EOF`. | open (D12) | Device memory is limited (256 MB on emunpu). |
| R7 | Every jump target is a valid instruction number. | implemented | The runtime rejects an out-of-range target. |
| R8 | Do not use `@N` on an id operand. | implemented | See [Operands](#3-operands). |
| R9 | A register is set before it is read. | implemented | The runtime rejects a read of an unset register. |
| R10 | `EOF` is the last instruction. | implemented | Ends the program clearly. |
| R11 | The file starts with `.version 1`, then the header, then the instructions. | implemented | The runtime checks the format version before it reads anything else. |
| R12 | Every declared input is read by a `COPY_H2D`. Every declared output is written by a `COPY_D2H`. The runtime checks this at `EOF`, so it also sees slots that use `@N`. | implemented | An unused declaration usually means a lowering bug. |
| R13 | No `LAUNCH` argument is negative. Every `LAUNCH` scalar obeys the range of its **kind**: a `DIM` is at least 1, a `FLAG` is 0 or 1, a `PARAM` is any value. See [Scalar kinds](#51-scalar-kinds). | implemented | A zero-size tensor cannot be allocated (R1), so a zero `DIM` is almost always a lowering bug. A `FLAG` is a mode, so 0 is valid. The verifier checks literals; the run checks `@N` values against the op's signature row. |

## 7. Limits

| Limit | Value | Status |
|---|---|---|
| Registers | ids 0 to 1023 | implemented |
| Input slots | 0 to 255 | implemented |
| Output slots | 0 to 255 | implemented |
| Live buffers | 4096 (or the device's cap, if lower) | implemented |
| Single `ALLOC` | 256 MB (emunpu device memory) | implemented (by the device) |
| Total live device memory | 256 MB | implemented (by the device) |
| Instructions run in one program | 100,000,000 | implemented |
| Operand value | signed 32-bit | implemented |

If a model needs a larger limit, tell the runtime team. The limits are configuration values, not fixed design.

## 8. What the runtime guarantees

- The runtime stops at the first error. It does not read outputs after an error.
- Each error has a message and the line number in the bytecode file.
- The program exits with a non-zero code on error.
- An error is reported, not a crash, for any input that breaks the rules in [§6](#6-rules-the-compiler-must-follow) and that the runtime checks.
- When the planned load-time checks are done, an error that is visible in the file text is reported before any device work starts.

## 9. Open items

These need a decision from both teams before version 1.0.

| # | Question | Runtime proposal |
|---|---|---|
| O1 | Header format | **Proposed (§2):** `.input <slot> <bytes> <name>` and `.output <slot> <bytes> <name>`. Add dtype and shape later. |
| O2 | Buffer ids | Ids by `ALLOC` order (as today). Later, possibly `ALLOC <reg> <size>`, with the id put in a register (as in IREE). |
| O3 | Who frees buffers | The compiler emits every `FREE`. The runtime reports leaks (strict in tests). |
| O4 | Who tiles a large tensor | The compiler plans tiles and memory. The runtime gives offset copies and checks. |
| O5 | Text or binary bytecode | Text until the pipeline runs end to end. Binary later, if load time matters. |
| O6 | The downsized Llama-3 configuration | Needed to set the limits and the operator list. |
| O7 | Version line | **Proposed (§2):** `.version 1` is the first line. The runtime rejects any other version. |
| O8 | Which weight goes in each slot | **Proposed (§2):** the name is on each header line, for example `.input 3 1048576 layers.0.wq`. A separate manifest was rejected: two files can disagree. |
| O9 | One op id per activation, or one `ACTIVATION` op id with the kind as a scalar? | **Decided 2026-11-21: one shared `ACTIVATION` op id** (op id 7) taking `count, kind, alpha_bits, beta_bits` as scalars, `kind` a new `ENUM` scalar range-checked against the 24 kinds `emunpu_activation` supports, `alpha_bits`/`beta_bits` a new `FLOAT_BITS` scalar kind always present (kinds that ignore them pass the bits of `0.0f`). Adds every activation at once instead of one op id each. RELU keeps op id 2 and its own LAUNCH shape (`count` only), but `do_relu` is now a thin wrapper around the same `emunpu_activation` call ACTIVATION uses (`kind` fixed to RELU, `alpha`/`beta` defaulted to `0.0` instead of read from `args`) — one implementation, two LAUNCH spellings, not two code paths. The signature row, dispatch wiring and `do_activation`'s body (including the `FLOAT_BITS` encode/decode) are fully implemented. |

## 10. Example

A program that adds two vectors of 4 floats. The header is described in [§2](#2-header-version-inputs-and-outputs).

```
.version 1
.input  0 16 a
.input  1 16 b
.output 0 16 c

ALLOC 16              // 0: buffer 0 (a)
ALLOC 16              // 1: buffer 1 (b)
ALLOC 16              // 2: buffer 2 (c)
COPY_H2D 0 0 16       // 3
COPY_H2D 1 1 16       // 4
LAUNCH 1 3 0 1 2 1 4  // 5: VEC_ADD, c = a + b, 4 elements
COPY_D2H 0 2 16       // 6
FREE 0                // 7
FREE 1                // 8
FREE 2                // 9
EOF                   // 10
```
