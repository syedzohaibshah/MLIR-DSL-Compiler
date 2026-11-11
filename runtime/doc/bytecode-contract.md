# Bytecode contract: compiler → runtime

**Version:** 0.2 (draft for agreement)
**Date:** 2026-11-02
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
| 2 | RELU | `A, C` | `count` | `C[i] = max(0, A[i])` for `count` elements |
| 3 | BIAS_ADD | `A, bias, C` | `count, out_features` | `C[i] = A[i] + bias[i mod out_features]` for `count` elements |

The Llama-3 operators (RMS norm, SiLU, softmax, RoPE, gather, transpose, element-wise multiply) are not available yet. Their op ids and operands will be added to this table by agreement.

## 6. Rules the compiler must follow

| # | Rule | Status | Reason |
|---|---|---|---|
| R1 | An `ALLOC` size is greater than 0 and is a multiple of 4. | implemented | emunpu stores every buffer as fp32. |
| R2 | A copy size **equals** the declared slot size. A partial copy is an error. | implemented | Keeps the binding simple. Partial copies come later, with copy offsets for tiling. |
| R3 | A copy size is not larger than the buffer size. | implemented | Stops a copy past the end of a buffer. |
| R4 | A LAUNCH reads and writes only inside its buffers. For example, for MATMUL: `M·K·4 ≤ size(A)`, `K·N·4 ≤ size(B)`, `M·N·4 ≤ size(C)`. | planned | The device does not stop a kernel that writes past a buffer. |
| R5 | Do not use a buffer after `FREE`, and do not free it twice. | implemented | The runtime rejects it. |
| R6 | Free every buffer before `EOF`. | open (D12) | Device memory is limited (256 MB on emunpu). |
| R7 | Every jump target is a valid instruction number. | implemented | The runtime rejects an out-of-range target. |
| R8 | Do not use `@N` on an id operand. | implemented | See [Operands](#3-operands). |
| R9 | A register is set before it is read. | implemented | The runtime rejects a read of an unset register. |
| R10 | `EOF` is the last instruction. | implemented | Ends the program clearly. |
| R11 | The file starts with `.version 1`, then the header, then the instructions. | implemented | The runtime checks the format version before it reads anything else. |
| R12 | Every declared input is read by a `COPY_H2D`. Every declared output is written by a `COPY_D2H`. The runtime checks this at `EOF`, so it also sees slots that use `@N`. | implemented | An unused declaration usually means a lowering bug. |

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
