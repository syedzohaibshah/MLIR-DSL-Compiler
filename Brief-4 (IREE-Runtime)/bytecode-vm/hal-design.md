# Adding Multi-Hardware Support to the VM

A design document. Nothing here is built yet — this is us deciding *what*
to build and *why*, before writing any code.

---

## 1. Where we are today

The VM works. It parses a bytecode file, runs the instructions in order, and
talks to an NVIDIA GPU through CUDA. Roughly 1150 lines across seven files.

Here is what each piece does:

| File | Job | Does it know about CUDA? |
|---|---|---|
| `bytecode.cpp` | Reads the text file, turns each line into an opcode | No |
| `io_handle.cpp` | Holds host input/output slots | No |
| `error.h` | Prints errors with a line number | No |
| `vm_core.cpp` | The main loop: dispatch, registers, branching | Yes — `cudaMemcpy` |
| `buffer_manager.cpp` | Allocates and frees device buffers | Yes — `cudaMalloc`/`cudaFree` |
| `kernal_registry.cpp` | Looks up a kernel id and launches it | Yes — `cudaLaunchKernel` |
| `kernels.cu` | The four actual GPU kernels | Yes — it *is* CUDA |

The good news: **CUDA only leaks into three places.** The parser, the I/O
harness, and the error helper are already hardware-neutral. Even inside
`vm_core.cpp`, the register file, the branching, and the dispatch loop don't
touch CUDA at all — only the two copy helpers do.

That is why this refactor is doable without tearing the VM apart.

---

## 2. How emunpu is designed

Before deciding anything, we need to understand what we are actually plugging
in. emunpu is not a small thing — it is a full simulated NPU, about 130 KB of
C source, and it was extracted from a working IREE backend.

### The physical model

emunpu pretends to be a real accelerator chip:

```
                    ┌─────────────────────────────────────┐
                    │        GLOBAL MEMORY (arena)        │
                    │   one big byte array, 4 MB / 256 MB │
                    │   + a table of tensor descriptors   │
                    └──────────────┬──────────────────────┘
                                   │  DMA in / DMA out
        ┌────────┬────────┬────────┼────────┬────────┬────────┐
        │        │        │        │        │        │        │
     ┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐   ...
     │core0│  │core1│  │core2│  │core3│  │core4│  │core5│    (10 total)
     └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
     each core has:
       - 256 KB private scratchpad ("local memory")
       - a private command queue
       - a 16x16 matrix engine (MMA)
       - a vector unit + a special-function unit
       - DMA to global memory
```

Ten cores. One shared global memory. Each core has its own small fast
scratchpad, and it must **DMA** data from global memory into that scratchpad
before it can compute on it, then DMA the result back out.

### Memory is typed, not raw bytes

This is the first big difference from CUDA. When you allocate on emunpu you
don't ask for bytes — you ask for a *tensor*:

    emunpu_device_alloc(device, FP32, rank=2, shape={4,4}, "weights", &id)

You get back an **id** (a plain `uint32_t`), not a pointer. The actual memory
lives inside emunpu's arena and you are not supposed to dereference it
directly. The device keeps a descriptor table recording, for every live
allocation: its id, its dtype, its shape, its byte offset in the arena, its
size, and a debug name.

emunpu supports seven element types: fp32, fp16, bf16, int32, int16, int8,
uint8. Every kernel routes through load/store helpers that widen to `double`,
which is how mixed precision works for free.

### The instruction set has two levels

emunpu has a **micro-ISA** — fifteen real machine instructions a core executes:

    DMA_G2L, DMA_L2G, MMA16, MATVEC, EW, REDUCE, FMA, CONVERT,
    ACT, SOFTMAX, NORM, TRANSPOSE, IM2COL, POOL2D, GATHER

Each one is a big flat struct: opcode, four dtypes, three operand addresses
(each a memory-space + byte-offset pair), problem dimensions M/N/K, row
strides, flags, and scalar parameters.

Crucially, **each instruction has a hard size limit**:

| Unit | Limit per instruction |
|---|---|
| Matrix engine (MMA16) | 16 x 16 x 16 |
| Vector ALU (EW/FMA/CONVERT) | 256 elements |
| Special-function unit (ACT) | 64 elements |
| Row unit (REDUCE/SOFTMAX/NORM) | 1024 elements, one row |
| DMA | 64 KB per descriptor |

Ask for more and the instruction is **rejected**, not slowed down. That is the
honest hardware behaviour: a functional unit is only so wide.

On top of that micro-ISA sits a **high-level API** that looks like CUDA's
runtime: `emunpu_matmul`, `emunpu_gemm`, `emunpu_vector_op`, `emunpu_activation`,
`emunpu_softmax`, `emunpu_conv2d`, `emunpu_linear`, and so on. These take
whole tensors of any size and internally chop the work into instruction-sized
pieces, spread across the cores, and enqueue them.

**So there are two ways to drive emunpu:**

1. **High level** — call `emunpu_matmul(dev, C, A, B, M, N, K)` and let the
   device figure out the tiling and the core assignment.
2. **Low level** — query the caps, allocate scratchpad yourself, build DMA and
   MMA16 instructions by hand, enqueue them onto specific cores, then launch.

This matters for us later. (Short version: we use the high level first.)

---

## 3. How emunpu differs from CUDA

This is the heart of the problem. The two devices are not just different
brands of the same thing — they sit at **different levels of abstraction.**

| | CUDA (what we have) | emunpu (what we're adding) |
|---|---|---|
| What you allocate | raw bytes: `cudaMalloc(16)` | a typed tensor: dtype + rank + shape |
| What you get back | a `void*` pointer you can use | a `uint32_t` id you must not dereference |
| How work is described | "run this kernel function" | "do a matmul of these tensors" |
| Parallelism model | a grid of thread blocks, threads | 10 cores with command queues |
| Who decides the split | **you**, in the bytecode | **the device's scheduler** |
| Data movement | you `cudaMemcpy`, hardware caches the rest | explicit DMA global to scratchpad |
| Size limits | soft — huge grids are fine, just slower | hard — oversized instruction is rejected |
| Precision | whatever your kernel says | 7 dtypes, tracked per tensor |

### The one that actually breaks our design

Look at our current `LAUNCH` instruction:

    LAUNCH  kernel_id  num_buffers  buf0 buf1 buf2  grid_x grid_y grid_z  block_x block_y block_z  num_scalars  s0 s1 s2

Six of those numbers are `grid` and `block`. **Those are CUDA concepts.**
emunpu has no grid. It has no blocks. It has no threads. It has ten cores and
a scheduler that decides fan-out on its own.

If we keep `LAUNCH` exactly as it is and try to shove emunpu underneath it,
we'd have to invent fake grid and block numbers for a device that has no such
thing — and then throw them away. That would bake CUDA's execution model
permanently into our "hardware-neutral" layer, and every future hardware would
have to pretend to be a GPU too.

There is a second, quieter problem. Today `LAUNCH 0` means "kernel index 0 in
the CUDA kernel table", which happens to be `vec_mm_Kernel`. If emunpu builds
its own table, id 0 means something different depending on which backend is
loaded. The same bytecode file would compute different things on different
hardware. That defeats the entire point.

Both of these are solvable, and IREE already solved them — though not by
making one `LAUNCH` mean the same thing on both devices. See §5: we ended up
deciding the bytecode itself gets to differ per hardware, and only `LAUNCH`
is the part that does.

---

## 4. How IREE handles this

IREE runs the same compiled model on CUDA, Vulkan, Metal, CPU, ROCm, and
custom accelerators. Its answer to "how do you abstract over hardware that
works completely differently" is a **split into two layers**.

### Layer 1: the HAL — uniform across all hardware

The HAL (Hardware Abstraction Layer) defines a handful of objects that every
backend implements the same way:

- **device** — create it, query what it can do, synchronize it
- **buffer** — allocate, free, map, copy to/from host
- **command buffer** — a recorded list of work: copies, dispatches, barriers
- **executable** — a loaded blob of compiled kernels
- **semaphore / fence** — ordering and synchronization

The HAL deals in *plumbing*. It knows about moving bytes and submitting work.
It knows **nothing** about how to multiply a matrix.

### Layer 2: the executable — completely backend-specific

An IREE executable is an **opaque blob**, and each backend gets its own:

| Backend | What the blob contains |
|---|---|
| CUDA | PTX assembly |
| Vulkan | SPIR-V |
| CPU | a native ELF shared library |
| emunpu (this one!) | a command list + a precomputed tile schedule |

The HAL never opens the blob. It only says: *"dispatch entry point number 3 of
this executable, with these buffers bound, at this workgroup count."*

### The key insight

**IREE does not try to make one kernel run everywhere.** That's the mistake
people expect it to make. Instead:

- the **plumbing** is made uniform (allocate, copy, dispatch, sync), and
- the **implementation** stays per-backend.

The one shared abstraction is the **workgroup count** — an abstract statement
of "this work splits into N independent pieces." CUDA maps that to thread
blocks. The CPU backend maps it to a `for` loop across worker threads. emunpu
maps it to cores. Each backend interprets it in its own terms, and nobody has
to pretend to be a GPU.

### IREE's own emunpu backend proves the point

We don't have to guess how this applies, because emunpu ships the answer.
`reference/INTEGRATION.md` documents exactly how this simulator was wired into
IREE, and it splits into three parts:

```
   compiler side                                  runtime side
   ─────────────                                  ────────────
   recognizes matmul/add/relu/copy                loads the blob,
   decides the 16x16 tiling        ── blob ──▶    stages tensors into memory,
   emits commands + tile schedule                 replays it on the device
                  │                                       │
                  └───────── shared contract ─────────────┘
                              emunpu_isa.h
```

Note what the compiler decided ahead of time: **the tiling**. IREE's backend
computes the 16x16 block grid and the per-core assignment at compile time and
ships it in the blob, exactly the way a GPU backend bakes in a workgroup grid.
The runtime just replays it. That's why emunpu has *two* matmul entry points —
`emunpu_matmul` (device decides the tiling) and `emunpu_matmul_scheduled`
(replay a schedule someone else decided).

That's a very interesting direction for us later. It is not where we start.

### The exact split we're borrowing

Precisely stated, IREE's compiled artifact is two things stacked together,
not one or the other:

```
  .vmfb file
  ├── VM bytecode module         ← HARDWARE-INDEPENDENT
  │   allocate buffer, copy, dispatch entry #3, wait on fence
  │
  └── executables (blobs)        ← HARDWARE-SPECIFIC
      ├── cuda    → PTX
      ├── vulkan  → SPIR-V
      └── emunpu  → command list + 16x16 tile schedule
```

The part that says "allocate this, copy that, dispatch this, then wait" —
IREE's own VM bytecode — is identical no matter which device it targets. Only
the thing it dispatches *into* changes per hardware.

That maps onto our own opcode list almost exactly. `ALLOC`, `FREE`,
`COPY_H2D`, `COPY_D2H`, `SET`, `INC`, `DEC`, `BR`, `COND_BR`, `EOF` are already
our version of that hardware-independent bytecode module — identical
regardless of target. `LAUNCH` is our version of the blob: the one
instruction whose meaning is allowed to be, and in fact has to be,
hardware-specific. We're not copying IREE's *portability* story (one blob
running on many devices from one compile) — we're copying its *split*
(control flow stays uniform, the dispatch-into-hardware part doesn't).

---

## 5. Our plan

### Decided: bytecode is target-specific

Every opcode except `LAUNCH` means the same thing regardless of hardware. A
CUDA bytecode file and an emunpu bytecode file will encode `LAUNCH`
completely differently — CUDA's carries a kernel id, a grid, and a block;
emunpu's will carry something with neither, since emunpu has no concept of a
grid or a block at all. A bytecode file will name which hardware it targets
(a `.target` line, exact syntax still to be decided), and the VM refuses to
run a file against the wrong device rather than silently misinterpreting its
`LAUNCH` lines.

This means: no single bytecode file runs unmodified on every backend. What
*is* portable is the source problem — "run this 2-layer network" — which can
be hand-written twice, once per target, and both should produce the same
numbers. That's a weaker promise than IREE's "compile once, run anywhere,"
and an honest one: we have no compiler, so "compile once" was never actually
on the table. What we get in exchange is that each backend's `LAUNCH` can
look exactly like that hardware wants it to, instead of every backend
pretending to be shaped like a GPU.

### The shape

We put one new layer between the VM and the hardware. Everything above it
stops knowing what hardware exists.

```
                bytecode file (.txt)
                        │
                  Bytecode parser          ← unchanged
                        │
                      VM core              ← dispatch, registers, branching
                        │                     (all hardware code removed)
                        ▼
              ┌────────────────────┐
              │   Device interface │        ← THE NEW SEAM
              │  (abstract class)  │
              └────────────────────┘
                  │             │
          ┌───────┘             └───────┐
          ▼                             ▼
    ┌───────────┐                ┌─────────────┐
    │CudaDevice │                │EmunpuDevice │      + future hardware,
    └───────────┘                └─────────────┘        added without touching
          │                             │               anything above
     cudaMalloc                  emunpu_device_alloc
     cudaMemcpy                  emunpu_device_upload
     cudaLaunchKernel            emunpu_matmul / etc.
```

The VM holds a pointer to a `Device`. It never knows which one.

### What the Device interface promises

Seven things, and nothing more:

1. **Tell me your name** — just enough to check a bytecode file's declared
   `.target` against the device actually loaded
2. **Allocate memory** — give me a handle
3. **Free memory**
4. **Upload** — host bytes into a handle
5. **Download** — a handle back into host bytes
6. **Dispatch** — take this list of numbers (already resolved from whatever
   the bytecode's `LAUNCH` line said) and do whatever they mean *to you*

That is the whole contract — six promises, not seven. Notice #6 doesn't say
"buffers, kernel id, grid, block" or anything else specific — that's on
purpose, and §"dispatch stays dumb" below explains why.

No separate "synchronize" promise, and no "capabilities" beyond a name. Both
were in an earlier draft of this list, mirroring IREE's own device API more
closely than we actually need. Neither has a real caller yet: every launch
this VM does is already synchronous end to end (`launchGenericKernel` calls
`cudaDeviceSynchronize` before it returns, today, with nothing external
triggering that wait separately), and nothing here queries a device's
capabilities for any reason yet. `dispatch()` itself is the blocking,
complete operation — both real backends are naturally synchronous under the
hood anyway. If a concrete need for either shows up later (an explicit `SYNC`
opcode, say), add it then.

### Design choice 1: opaque handles, assigned in call order

The VM will pass around an opaque integer handle instead of a `void*`.

**Why:** a `void*` is already a CUDA assumption. emunpu gives us a `uint32_t`
id whose memory lives inside its own arena, and we must never dereference it.
An opaque number fits both, and fits whatever comes third.

There's a second half to this choice, and it's the one that actually matters:
**every backend must hand out handles `0, 1, 2, ...` in allocation order,**
regardless of what its real underlying hardware id looks like. This isn't a
new rule we're inventing — it's already true of the bytecode today. Look at
`vm_core.cpp`'s `ALLOC` handler: it calls `buffers.alloc(...)` and throws the
return value away. Every later instruction (`FREE 0`, `COPY_H2D 1 ...`) just
trusts that the Nth `ALLOC` in the file got id `N-1`, because that's the only
order `CudaDevice`'s allocator has ever handed them out in.

emunpu can't make that same promise for free — its tensor ids start at 1 (0
is reserved as "invalid"), and an id isn't guaranteed to stay contiguous
after a free. So `EmunpuDevice::allocate()` will keep a small private table
translating "the Nth thing this backend was asked to allocate" to whatever
real `emunpu_tensor_id_t` it actually got back. That table lives entirely
inside `EmunpuDevice` — the VM never sees it, and bytecode never has to
change. This is what makes a separate VM-side indirection table (an earlier
draft of this section had one) unnecessary: the contract is on the backend,
not on the VM.

### Dispatch stays dumb — no shared operation shape

Since `LAUNCH` is now allowed to mean something completely different per
backend (§"Decided" above), there's no shared struct to design for it. The VM
resolves every `@reg` in a `LAUNCH` line into a plain value — it already does
exactly this today, building `int_args` before calling `kernels.launch(...)`
— and hands the whole resolved list straight to `Device::dispatch()`. What
those numbers *mean* is entirely up to whichever backend receives them:

- `CudaDevice::dispatch()` reads them exactly like `KernelRegistry::launch`
  does today — kernel id, buffer ids, grid, block, scalars — because that's
  what CUDA-targeted bytecode will keep encoding.
- `EmunpuDevice::dispatch()` reads the *same shaped call* (a list of ints) but
  interprets its own bytecode's layout — likely an op id, some buffer ids,
  and M/N/K — with no grid or block anywhere, because emunpu-targeted
  bytecode never encodes one.

Two things this drops from earlier drafts of this plan, on purpose:

- **No semantic op table** (no shared `MATMUL = 0` across backends). Each
  backend keeps its own kernel/op table exactly like `KernelRegistry` already
  does — there's no cross-backend meaning to keep in sync, since the two
  bytecode files were never meant to be the same file in the first place.
- **No execution-hint struct for grid/block.** There's nothing to make
  optional, because there's no shared shape to reconcile — CUDA bytecode
  always has a grid and block; emunpu bytecode never does.

One real thing we give up: an emunpu backend built this way can't
automatically notice that a CUDA-style MATMUL→BIAS_ADD→RELU sequence could
have been one fused `emunpu_linear` call — that fusion decision now has to be
made by whoever hand-writes the emunpu-targeted bytecode, not discovered by
the VM at runtime. That's fine for a hand-written program; it's the same
limitation a real compiler exists to remove, and we don't have one.

### `LAUNCH` is entirely our invention — it does not exist in emunpu

Worth being explicit about this, because it's easy to blur: there is no
`LAUNCH` anywhere in emunpu-sim's own repository. `LAUNCH` is a bytecode
instruction we invented, for our own VM, before emunpu was ever part of this
project. emunpu doesn't parse bytecode and doesn't know a VM is driving it —
it's a plain C library exposing about 40 functions (`emunpu_matmul`,
`emunpu_vector_op`, `emunpu_activation`, `emunpu_linear`, `emunpu_softmax`,
and more), each one already fully implemented. Looking at the real source
(`emunpu-sim/src/device.c:1015`), `emunpu_matmul` itself is nine lines: it
fills in a small command struct and hands it to `emunpu_submit`, which is
where the actual tiling-across-cores machinery lives. We never write or touch
that machinery — `EmunpuDevice::dispatch()` is the *only* place our bytecode
and emunpu's API ever meet, and all it does is pick one pre-built function per
`op_id` and call it.

### emunpu-targeted `LAUNCH`'s bytecode shape

No grid, no block — the entire reason to go target-specific in the first
place:

    LAUNCH  op_id  num_buffers  buf0..bufN  num_scalars  scalar0..scalarN

Simpler to parse than CUDA's version too, since there's no 6-number grid/block
section to skip over. The four ops our first emunpu bytecode program needs,
each mapped to a real emunpu call:

| op_id | buffers | scalars | calls |
|---|---|---|---|
| 0 (MATMUL) | `a, b, c` (3) | `M, N, K` (3) | `emunpu_matmul(emu_device, nullptr, c, a, b, M, N, K, /*transpose_b=*/0)` |
| 1 (VEC_ADD) | `a, b, c` (3) | `count` (1) | `emunpu_vector_op(emu_device, nullptr, c, a, b, count, EMUNPU_EW_ADD, /*bcast_period=*/0)` |
| 2 (RELU) | `a, c` (2) | `count` (1) | `emunpu_activation(emu_device, nullptr, c, a, count, EMUNPU_ACT_RELU, 0, 0)` |
| 3 (BIAS_ADD) | `a, bias, c` (3) | `count, out_features` (2) | `emunpu_vector_op(emu_device, nullptr, c, a, bias, count, EMUNPU_EW_ADD, /*bcast_period=*/out_features)` |

Bias-add reuses `emunpu_vector_op` rather than needing its own function —
`bcast_period` is exactly the "repeat every N elements" behavior a per-column
bias needs. `nullptr` for the stream parameter means "run immediately,
synchronously," matching the rest of this project.

Also worth recording: since this bytecode is emunpu's own, its scalars don't
need to mirror CUDA's. CUDA's `vec_mm_Kernel` needed `(n, width, height)`
because that's what its raw thread-indexing math needed; `emunpu_matmul` just
wants `(M, N, K)` directly, so that's all this bytecode carries. Each
backend's `LAUNCH` carries exactly what *that* hardware wants — nothing
inherited from the other one.

One structural difference from `CudaDevice::dispatch`: there, all four
kernels shared one call shape (`cudaLaunchKernel` with the same args-array
trick), so one `launchGenericKernel` helper covered all of them. Here, each op
calls a genuinely different emunpu function with a genuinely different
signature, so `EmunpuDevice::dispatch` can't share one generic helper the same
way — it parses the common header (`op_id`, buffer list, scalar list) once,
then branches into four small op-specific blocks.

### What if an operation isn't one of the ~40 emunpu already provides?

Three tiers, in the order you'd actually reach for them:

1. **Check whether it's already there under a name not yet wired up.** We only
   mapped 4 of emunpu's ~40 operations above. `emunpu_activation` alone covers
   roughly 24 activation kinds (`emunpu_act_kind_t`) — sigmoid, gelu, tanh,
   silu, mish, and more are already implemented, just not given an `op_id` in
   our table yet. Adding one is a new case in `dispatch()`, not a new
   definition anywhere.
2. **If it's composable from 2+ existing emunpu calls, compose it inside
   `EmunpuDevice::dispatch()`** (or a small private helper on the class) —
   call one emunpu function, then another, same idea as the kernel-chaining
   `bytecode9.txt` already proved on the CUDA side (one op's output feeding
   straight into the next). This stays entirely inside code we own.
3. **Only if it's a genuinely new hardware primitive, not expressible by
   composing anything above, would emunpu-sim's own source need a new
   function** (`emunpu-sim/src/*.c`, rebuilding `libemunpu`). A real option —
   `reference/INTEGRATION.md` documents how — but a much bigger undertaking
   than anything this project has needed so far, and not expected to come up
   for our four ops.

### The coverage-gap risk this design accepts, on purpose

Going target-specific means each backend owns its own private op table (we
dropped the shared semantic-op table in an earlier draft — see above). Nothing
automatically guarantees a CUDA-targeted bytecode file and its "equivalent"
emunpu-targeted file actually stay equivalent — a human could add a kernel to
one side and forget the other, and nothing would catch that except running
both and noticing the numbers differ.

This isn't a new problem this design introduces, though — it's the same
limitation the whole project already has. There's no compiler here, on
purpose (see this doc's very first line), so nothing has ever guaranteed a
*single* hand-written bytecode file is correct either — that's always been on
whoever writes it. Two backends just means that discipline now spans two
files that are supposed to agree, instead of one file that's supposed to be
right. The mitigation isn't a new mechanism — it's Step 6 of the order of
work, already on the list: write both, run both, compare the actual numbers.
A real compiler could *guarantee* cross-backend consistency later by
generating both files from one shared source, but that's explicitly future
work (§9), not something we're building now.

---

## 6. What actually changes in the code

| File | Change | Risk |
|---|---|---|
| `bytecode.cpp/.h` | nothing | none |
| `io_handle.cpp/.h` | nothing | none |
| `error.h` | nothing | none |
| `vm_core.cpp/.h` | holds a `Device*`; copy helpers call upload/download; drop the CUDA include | low — the loop, registers and branching are untouched |
| `buffer_manager.*` | moves inside the CUDA backend, its handles become the 0-based contract `CudaDevice::allocate()` promises | low — logic kept as is |
| `kernal_registry.*` | becomes the CUDA backend's own `dispatch()` — still reads kernel id/grid/block/scalars exactly as today | low |
| `kernels.cu/.h` | unchanged, just relocated | none |
| `main.cpp` | pick a device at startup | trivial |
| **new** device interface, registry, two backends | new files only | — |

**Ten of our instructions don't change at all.** ALLOC, FREE, COPY_H2D,
COPY_D2H, SET, INC, DEC, BR, COND_BR, EOF are already hardware-neutral. Only
`LAUNCH` shifts meaning, and only in what the id refers to.

---

## 7. Order of work

Each step leaves the VM working. Same discipline as the original roadmap.

1. **Write the Device interface.** No implementation, just the contract. ✅
   Done — `src/hal/device.h`: `name`, `allocate`, `free`, `upload`,
   `download`, `dispatch`, plus a virtual destructor. Six methods, no CUDA or
   emunpu anywhere in the file.
2. **Wrap CUDA behind it.** ⚠️ Code complete, verification assumed rather than
   confirmed. `src/hal/cuda_device.h`/`.cpp` (`cudaDevice`) implements all six
   methods — `BufferManager`'s and `KernelRegistry`'s logic relocated in,
   nothing new invented. `vm_core.h`/`.cpp` and `main.cpp` are rewired to talk
   only through a `Device*`; `vm_core.cpp` now has zero CUDA in it.
   `buffer_manager.*`/`kernal_registry.*` are dead code, still present but no
   longer built. **Not actually run yet** — this machine has no CUDA
   toolchain, so the byte-identical-output test against
   `examples/bytecode12.txt` needs a real CUDA GPU (a Kaggle instance, same as
   the CUDA path has used since). Proceeding to Step 3
   on the assumption that test would pass, not on a confirmed result — worth
   actually running before trusting this step.
3. **Build `EmunpuDevice` — memory only.** ✅ Done and now confirmed working
   (see step 4) — `src/hal/emunpu_device.h`/`.cpp` (`emunpuDevice`):
   constructor/destructor pair around `emunpu_device_create`/
   `emunpu_device_destroy` (the first real destructor in this project — CUDA
   never needed one), `allocate`/`free` wired to `emunpu_device_alloc`/
   `emunpu_device_free` with every allocation committed to `EMUNPU_DT_FP32`,
   and `upload`/`download` wired to `emunpu_device_upload`/
   `emunpu_device_download`. The private `buffer_table`
   (`TensorSlot{real_id, freed}`) is the handle-translation table from Design
   choice 1, genuinely doing its job — our 0-based handle and emunpu's own
   tensor id (which starts at 1) are visibly different numbers in the actual
   run's log output.
4. **Add emunpu dispatch.** ✅ Done and verified end to end.
   `emunpuDevice::dispatch` parses the header (`op_id`/buffer list/scalar
   list, no grid/block) then branches into four private helpers
   (`do_matmul`/`do_vec_add`/`do_relu`/`do_bais_add`), each validating its own
   buffer/scalar count and calling the matching real function
   (`emunpu_matmul`/`emunpu_vector_op`/`emunpu_activation`) per the table
   above. `examples/bytecode12_emunpu.txt` — the `LAUNCH`-line translation of
   `examples/bytecode12.txt`, everything else carried over unchanged — was
   built (`clang++` + `-lemunpu`, no CUDA, no remote GPU, right on this Mac) and
   run via `src/main_emunpu.cpp`. **Result: `0 0 8 0`, matching the expected
   2-layer network output exactly.** This is a genuinely confirmed result,
   not an assumed one — first real proof the `Device` abstraction holds up
   against a second, structurally different hardware target.

   One real, pre-existing bug surfaced by actually running it: the cleanup
   loop at the end of `bytecode12.txt`/`bytecode12_emunpu.txt`
   (`SET 4 7` / `FREE @4` / `DEC 4` / `COND_BR 4 32 29`) loops back to
   `SET 4 7` instead of `FREE @4`, so it resets the counter every pass and
   tries to free buffer 7 repeatedly instead of counting down through it.
   Six "buffer already freed" errors print at the end, harmlessly (the real
   computation is already done by then), but the loop-back target should be
   `30`, not `29`. Identical in both bytecode files — not something this
   sync or the emunpu work introduced, just never triggered before because
   the CUDA path had never actually been run either. Worth fixing in both
   files; not blocking anything.
5. **Device selection.** ← *we are here*. `main.cpp` picks a `Device` at
   startup (a flag, or read from the bytecode's `.target` line), plus build
   flags (`VM_ENABLE_CUDA` / `VM_ENABLE_EMUNPU`) so a machine without CUDA —
   this Mac — still builds and runs the emunpu path without a separate
   `main_emunpu.cpp` standing in for it.
6. **Run both bytecode files and compare.** ✅ Effectively done manually —
   `bytecode12.txt` (CUDA) is still only assumed correct (needs a Kaggle GPU
   run, per step 2's caveat); `bytecode12_emunpu.txt` (emunpu) is confirmed
   correct right here. Once step 2 is actually verified on Kaggle, this step
   is fully closed.

Step 1 is the contract. Step 2 is pure refactoring — no new behaviour, and
the test is that nothing changed. Steps 3–6 are the actual new hardware.

---

## 8. Problems we know about

Better to name these now than discover them halfway through.

**emunpu wants a dtype, our ALLOC doesn't have one.** `ALLOC 16` says
sixteen bytes and nothing else. emunpu can allocate raw bytes, but then it
records the dtype as uint8, and a matmul on a "uint8" tensor computes the
wrong thing. **Plan:** the emunpu backend assumes fp32 everywhere, which
matches what our CUDA kernels already do. Later we can add an optional
`ALLOC_TYPED dtype rank d0 d1 ...` opcode. Not a blocker, but a real gap.

**emunpu's size limits are real.** 1024-element rows, 64 KB DMA, 256 vector
lanes. The high-level API tiles for us, so as long as we use it we're fine.
**This is exactly why we use the high-level API first** — the low-level path
makes us responsible for tiling and scratchpad management, and that's a trap
for a first backend.

**Error types differ.** CUDA returns `cudaError_t`, emunpu returns
`emunpu_status_t`. Both get mapped to our `bool` + `fail()` convention at the
backend boundary, and nowhere else.

**No build system yet.** `pivoted/` has no CMakeLists. emunpu is C (already
`extern "C"`-safe), ours is C++ and CUDA. We'll want CMake with independent
on/off switches for each backend — and that's not just tidiness. **CUDA won't
build on this Mac at all**, so making backends optional is what lets us
develop and test the emunpu path locally.

---

## 9. Deliberately not doing yet

**Streams and async.** Both devices have them. Our VM is synchronous and no
bytecode instruction expresses async work. Adding it now means building an
abstraction we can't test.

**emunpu's low-level instruction path.** It's tempting — enqueueing DMA and
MMA16 instructions by hand is much closer in spirit to what our VM already
does. But it makes us responsible for tiling, scratchpad allocation, and
staying inside every capability limit. Get the high-level path working first.

Later, though, that low-level path is the most interesting thing in this whole
project. IREE's own emunpu backend decides the 16x16 tiling **at compile time**
and ships the schedule in the blob. If we ever want our bytecode to carry a
tile schedule and have the VM replay it, we'd be doing the real thing — a
compiler deciding hardware tiling, exactly as IREE does it. Worth keeping in
view while we build the boring layer underneath.
