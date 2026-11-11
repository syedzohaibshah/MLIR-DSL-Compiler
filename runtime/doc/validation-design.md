# Validation design: limits, binding, and checks

**Status:** adopted as decision D14 (2026-11-01). Steps 1, 2 and 3 done (2026-11-11). Step 4 (verifier) and step 5 (kernel shapes) are open. See [Migration plan](#9-migration-plan).

This document defines where the runtime checks each fact about a bytecode program: limits, slots, sizes, buffers, and operands. It is the reference for every new check. Before you add a check, find its owner in [the ownership table](#8-ownership-table).

Related documents:

- [`../doc/bytecode-vm-project.md`](../doc/bytecode-vm-project.md): how the VM works today.
- [`hal-design.md`](hal-design.md): the `Device` interface.
- [`../doc/bytecode-contract.md`](../doc/bytecode-contract.md): what the compiler team must emit.
- [`../context.md`](../context.md): open issues and the plan.

## Contents

1. [Problem](#1-problem)
2. [Principle](#2-principle)
3. [Architecture](#3-architecture)
4. [Part 1: RuntimeLimits](#4-part-1-runtimelimits)
5. [Part 2: Verifier](#5-part-2-verifier)
6. [Part 3: Signature](#6-part-3-signature)
7. [Part 4: BufferTable](#7-part-4-buffertable)
8. [Ownership table](#8-ownership-table)
9. [Migration plan](#9-migration-plan)
10. [Failure modes](#10-failure-modes)
11. [References](#11-references)

## 1. Problem

Checks are spread over four layers. Some facts are checked twice, and some facts are checked nowhere.

| Fact | Checked today in | Gap |
|---|---|---|
| Opcode arity, `@` on an id | each VM handler | repeated in 5 handlers |
| Copy size vs slot size | parser scan (`expectedSize`) and `checkInputSlot` | the scan cannot see `@N` slots; a conflict only prints |
| Copy size vs buffer size | emunpu device only | CUDA does not check |
| Buffer handle is live | device only | the VM does not know |
| ALLOC handle | returned by the device | the VM ignores it |
| LAUNCH shape vs buffer size | nowhere | a kernel can write into other tensors |
| Caps | `vm_core.h`, `io_handle.h`, `execute` | no cap on ALLOC size or buffer count |

There are three root causes:

1. The VM has no record of buffers, so it cannot check copies.
2. The slot sizes are guessed from the code, not declared.
3. Every check runs late, inside a handler. Nothing is checked once at load.

## 2. Principle

> **Check each fact once, in the earliest layer that can know it.**

- A fact visible in the file text is checked at **load time**.
- A fact that depends on a register (`@N`) is checked at **run time**.
- A fact that depends on the hardware is checked in the **device**.

## 3. Architecture

```
 RuntimeLimits ─────────────── caps for every layer ───────────────┐
                                                                   │
 file ─► Parser ─► Verifier ─► Signature bind ─► VM run ─► Device ◄┘
         syntax    static      host inputs       dynamic    hardware
                   facts       and outputs       facts      facts
```

| Layer | Runs | Knows |
|---|---|---|
| Parser | once, at load | text, integers |
| Verifier | once, after parse | operand kinds, literal values, the signature |
| Signature bind | once, before run | host pointers and sizes |
| VM | each instruction | register values, the `BufferTable` |
| Device | each device call | alignment, capacity, kernel semantics |

## 4. Part 1: RuntimeLimits

All caps move into one struct, `RuntimeLimits`, in `src/runtime_limits.h`. The VM gets it in its constructor.

The file is **not** named `limits.h`. A project file with that name shadows the C standard `<limits.h>`, because the compiler searches the including file's directory first. That breaks `CHAR_BIT`, `INT_MAX`, `<chrono>`, and `<string>`.

| Field | Default | Reason |
|---|---|---|
| `max_registers` | 1024 | Registers hold loop counters and indices. Lua uses 255. |
| `max_input_slots` | 256 | A downsized Llama with 16 layers needs about 150 weight slots. A full 32-layer Llama-3-8B needs about 290, so this value must be changeable. |
| `max_output_slots` | 256 | Same reason as inputs. |
| `max_live_buffers` | 4096 | Stops a runaway loop of ALLOCs. |
| `max_alloc_bytes` | min(device capacity, `INT_MAX`) | Asked from the device. emunpu gives 256 MB. |
| `step_limit` | 1e8 | Stops an infinite loop (D9). |

Reasons for a struct:

- One place to read and change every cap.
- A test can build a VM with small limits. For example, set `max_live_buffers = 4` and expect the 5th ALLOC to fail.
- IREE asks the device for its limits (`iree_hal_device_query_i64`) instead of hard-coding them.

### DeviceLimits

A second struct, `DeviceLimits` in `hal/device.h`, reports what the hardware allows. `Device::limits()` returns it.

| Field | emunpu value | Source |
|---|---|---|
| `max_allocation_bytes` | device capacity | `emunpu_device_global_mem(dev)->capacity` |
| `total_memory_bytes` | device capacity | same |
| `alignment_bytes` | 4 (fp32) | fixed dtype today |
| `max_live_buffers` | 1024 | `IREE_EMUNPU_MAX_TENSORS`; `0` means no limit |

`RuntimeLimits` is a policy the host sets. `DeviceLimits` is a fact the hardware reports. Where both apply, the stricter one wins — see [section 7](#7-part-4-buffertable).

`limits()` returns a plain struct, not an `Error`. A device whose constructor failed must therefore report something safe (all zero) rather than dereference a null handle.

## 5. Part 2: Verifier

The verifier walks the parsed program once, before any device call. One table drives it:

| Opcode | Operand kinds |
|---|---|
| `ALLOC` | VALUE |
| `FREE` | VALUE |
| `COPY_H2D` | VALUE VALUE VALUE |
| `COPY_D2H` | VALUE VALUE VALUE |
| `LAUNCH` | VALUE … (variadic) |
| `SET` | REG_ID VALUE |
| `INC`, `DEC` | REG_ID |
| `BR` | TARGET |
| `COND_BR` | REG_ID TARGET TARGET |
| `EOF` | none |

The verifier checks:

- The opcode name is known and the arity is correct.
- No `@` on a REG_ID or a TARGET (D11).
- A literal REG_ID is below `max_registers`.
- A literal TARGET is inside the program.
- A literal ALLOC size is above 0 and at most `max_alloc_bytes`.
- A literal COPY slot is declared in the signature, and a literal copy size equals the declared slot size.

Reasons:

- An error is reported before any device memory is used.
- A new opcode needs one table row, not new guards in a handler.
- Handlers become shorter. They resolve `@N` values and do the work.
- WebAssembly and the JVM validate a module once at load for the same reasons.

## 6. Part 3: Signature

The bytecode declares its inputs and outputs in header lines, before the first instruction:

```
.input  0 16
.input  1 64
.output 0 16
```

Format: `.input <slot> <bytes>` and `.output <slot> <bytes>`. Later: `.input <slot> <dtype> <shape>`, for example `.input 0 f32 4x4`.

| Who | Check |
|---|---|
| `setInput(slot, ptr, size)` | The slot is declared, and `size` equals the declared size. |
| Run start | Every declared input is set. |
| `COPY_H2D` (run time) | The slot is declared, and the copy size **equals** the slot size. |
| `COPY_D2H` (run time) | The slot is declared, and the copy size **equals** the declared output size. |
| `getOutput(slot, ptr, size)` | The slot is declared and written, and `size` equals the declared size. |

**Copy size rule:** for now, a copy size must **equal** the declared slot size. A partial copy is an error. Partial copies come later, together with copy offsets for tiling.

Reasons:

- The check works with `@N`, because the run-time check compares against the declaration, not against a scan of the file.
- One declaration replaces `expectedInputSize`, `expectedOutputSize`, `checkInputSize`, `checkOutputSize`, and `expectedSize`.
- IREE does the same: a compiled function carries its ABI signature, and the generated code calls `hal.buffer_view.assert` on each input.

The compiler team must emit these lines. See [`../doc/bytecode-contract.md`](../doc/bytecode-contract.md).

## 7. Part 4: BufferTable

The VM keeps one table: `handle → { size, live }`. It is the VM's record of every buffer.

### Implementation (`vm_core.h`, `vm_core.cpp`)

The table is a `std::vector<BufferSlot>`. **The vector index is the device handle.** Three private helpers own it:

| Helper | Job |
|---|---|
| `allocateBuffer(size, line)` | Check the live cap. Call `device->allocate`. Verify the returned handle. Append the slot. |
| `freeBuffer(handle, line)` | Call `checkBuffer`. Call `device->free`. Mark the slot dead. |
| `checkBuffer(handle, out_size, line)` | The shared lookup: the handle is in range and live. Gives back the size. |

`ALLOC` and `FREE` in `dispatch` call these instead of calling the device directly.

### Four design points

**1. A freed slot is marked dead, never erased.** Erasing from a vector indexed by handle shifts every later entry and silently breaks the mapping. A removed entry also cannot tell "already freed" apart from "never existed". Marking keeps both errors distinct.

**2. `live_buffer_count` is counted, not derived.** `buffers.size()` is how many buffers were *ever* allocated; it only grows. After 8 allocations and 8 frees, `size()` is 8 and the live count is 0. A loop over the slots could derive the count, but the counter is O(1).

**3. The append-only handle contract.** `Device` promises handles 0, 1, 2, … in call order, never reused. The VM depends on that promise, because `buffers[handle]` only works if the handle is the index. C++ cannot enforce this in the type system, so the contract lives in three places: the comment in `device.h`, the runtime check in `allocateBuffer`, and a unit test. The runtime check is the real enforcement — a device that returns pointers or recycles handles fails on the first `ALLOC` with a clear message, instead of writing to the wrong slot much later.

A device may still reuse the *memory* of a freed buffer. Only the handle — the name — stays unique. This mirrors IREE, where a ref-counted `iree_hal_buffer_t` sits above a pooling allocator.

**4. Two caps, not one.** `allocateBuffer` takes the smaller of `RuntimeLimits::MAX_LIVE_BUFFERS` and `DeviceLimits::max_live_buffers`. They answer different questions:

| Cap | Says |
|---|---|
| Device (emunpu: 1024) | "I cannot physically track more tensors." |
| VM (4096) | "A program that asks for this many buffers is broken." |

A device that reports `0` declares no limit of its own — a mock device in `unit_tests`, or a future CUDA device. Without the VM cap, a runaway `ALLOC` loop on such a device runs until the host is out of memory. The VM cap is the backstop for exactly that case.

### Why the size field is `int`

`BufferSlot::size` is `int` by decision D10. A single buffer is capped by the device (256 MB on emunpu), which fits an `int` easily.

If a future device allows a single buffer above 2 GB, this field **and** the size arguments on the copy path must become `int64_t` together. Changing only one truncates on assignment and hides the overflow instead of catching it.

A running **total** across buffers can already pass `INT_MAX` today, so any memory accounting added later must sum into `int64_t` even while this field stays `int`.

| Opcode | Run-time check |
|---|---|
| `ALLOC` | The live-buffer count is below `max_live_buffers`. The VM stores **the handle that the device returns**. |
| `FREE` | The handle exists and is live. The VM marks it not live. |
| `COPY_H2D` | The buffer is live, and the copy size ≤ the buffer size. |
| `COPY_D2H` | The buffer is live, and the copy size ≤ the buffer size. |
| `LAUNCH` | Every buffer operand is live. |
| End of run | The VM lists the buffers that are still live. The leak policy (D12) uses this list. |

Reasons:

- CUDA and emunpu reject the same bad program in the same way. Today emunpu rejects an oversize copy and CUDA overruns the buffer.
- Storing the returned handle fixes the binding issue (context.md B1, option A).
- Each check is O(1). The cost is small compared with a kernel.
- IREE stores `byte_length` in each `iree_hal_buffer_t`, and checks each copy with `iree_hal_buffer_validate_range`.

### What stays in the device

- Alignment (emunpu: size is a multiple of 4).
- Real capacity. The device is the final authority, because of fragmentation.
- Kernel shape vs buffer size, for example `M * K * 4 ≤ size of A`. Only the device knows what each `op_id` means. Calculate these products in 64-bit (D10 amendment).
- The handle-to-real-id translation.

### Why two tables, in the VM and in the device

The device also keeps the size of each buffer (add `size` to `TensorSlot`). This looks like duplication, but the two tables answer different questions:

| Table | Question | Why it must exist |
|---|---|---|
| VM | "Does this program make sense?" | The check runs once for every backend. Before this table, an oversize copy was rejected by emunpu but overran the buffer on CUDA. |
| Device | "Is this call safe?" | A device must stay safe when a caller uses it **without** a VM, as `unit_tests` does. A C API cannot assume its caller validated anything. |

IREE splits the same way: `iree_hal_buffer_validate_range` in the HAL, plus each driver's own checks.

## 8. Ownership table

| Fact | Owner | When |
|---|---|---|
| Syntax, integer parse, comments | Parser | load |
| Arity, operand kind, literal ranges | Verifier | load |
| Slot declared, slot size | Signature (literal: Verifier; `@N`: VM) | load and run |
| Register id range, INC/DEC overflow | VM | run |
| Buffer live, copy size ≤ buffer size | VM (`BufferTable`) | run |
| Caps | `RuntimeLimits` (+ device query) | load and run |
| Alignment, capacity, kernel shape vs buffer size | Device | run |
| VM handle to real device id | Device | run |

Rule D7 still holds: a device-independent check is in the VM; a device-specific check is in the device.

## 9. Migration plan

Do the steps in this order. Each step is small and has its own tests.

| # | Change | Closes | Needs | Status |
|---|---|---|---|---|
| 1 | `runtime_limits.h`: move the existing caps into `RuntimeLimits`; add `DeviceLimits` | scattered caps | — | done |
| 2a | `BufferTable` in the VM; store the ALLOC handle | double free, stale handle, B1 option A | 1 | done |
| 2b | Copy checks: `COPY_H2D` / `COPY_D2H` consult the table | copy size vs buffer size (both devices) | 2a | done |
| 3 | Signature `.input` / `.output`; delete the `expectedSize` scan | slot binding, `@N` blind spot | compiler team agreement | done (2026-11-11). Compiler team still to confirm the format. |
| 4 | Verifier and the opcode table; move the `@` guards out of the handlers | late errors, repeated guards | 1 | open |
| 5 | Device kernel-shape checks with buffer sizes | `gaps/err_launch_*` (D1) | 2b | open |

## 10. Failure modes

| Case | Result |
|---|---|
| The signature and a literal copy disagree | The verifier rejects the program at load. |
| The signature and an `@N` copy disagree | The VM rejects the instruction at run time. |
| The VM table and the device table disagree | The device error stops the run. The message is less clear, but no memory is corrupted. |
| Offset copies (future) | Check `offset + size ≤ buffer size` in 64-bit. In 32-bit, a large offset wraps around and passes. |
| Limits too small for real Llama | Change the `RuntimeLimits` default. No code change. |

## 11. References

- IREE buffer range check: `iree_hal_buffer_validate_range` in `runtime/src/iree/hal/buffer.c`.
- IREE input check: `hal.buffer_view.assert` in `compiler/src/iree/compiler/Dialect/HAL/IR/HALOps.td`.
- IREE device limits: `iree_hal_device_query_i64` in `runtime/src/iree/hal/device.h`.
- WebAssembly validation: [WebAssembly spec, Validation](https://webassembly.github.io/spec/core/valid/index.html).
- JVM verification: [JVM spec §4.10, Verification of class files](https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-4.html#jvms-4.10).
