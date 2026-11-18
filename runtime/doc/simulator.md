# emunpu-sim: what it is, and how we plug into it

**Date:** 2026-11-16
**Owner (runtime):** Syed Zohaib Shah
**Scope:** `emunpu-sim/` only. It is a given library. See rule 1 of the
project's own rules file: do not change it.

This file answers four questions:

1. What is `emunpu-sim`, in plain terms?
2. What do the `reference/` files do, and what is `emunpu_isa.h`?
3. Where does our runtime stand today against this simulator?
4. What do we have to build to integrate it properly?

---

## 1. What emunpu-sim is

`emunpu-sim` is a software model of a multi-core NPU. It is plain C11. It has
no dependency on IREE, Python, or any ML framework. You build it with `make`
or `cmake` and link `libemunpu.a` or `libemunpu.so`.

It models a real accelerator shape:

- **10 compute cores** (`IREE_EMUNPU_NUM_CORES`, a build-time constant).
- **One shared global memory** (an arena, default 256 MB). Buffers live here
  as typed tensors, not raw pointers.
- **Each core has its own local scratchpad** (256 KB), a command queue, a
  16×16 tensor-core matrix engine (MMA), a vector unit, and a DMA engine.
- **7 dtypes**: fp32, fp16, bf16, i32, i16, i8, u8.

Work moves in one direction only: global memory → DMA → a core's local
scratchpad → compute → DMA → global memory. The device API hides this for
you in most calls; the low-level API lets you do it by hand.

Execution has no timing model. It is bit-exact and synchronous: it exists to
check that a compiler and a runtime agree on what a program means, not to
predict how many cycles it takes.

### 1.1 Two API layers

| Layer | Header | What it is |
|---|---|---|
| High-level (device/runtime API) | `include/emunpu/device.h` | CUDA-like calls: alloc, upload, download, matmul, gemm, linear, softmax, layernorm, conv2d, activations, reductions. The call does the tiling for you. |
| Low-level (micro-ISA) | `include/emunpu/core.h`, `include/emunpu/arch.h`, `docs/MICRO_ISA.md` | Per-core instructions: `DMA_G2L`, `MMA16`, `DMA_L2G`, vector ops. You tile the problem and schedule cores yourself. |

A high-level call such as `emunpu_gemm` is implemented on top of the
low-level instructions — it builds the tile schedule and issues the
instructions for you. `docs/MICRO_ISA.md §8` shows the mapping.

### 1.2 What it supports (operations)

| Group | Operations |
|---|---|
| Matrix | matmul, gemm (+bias +relu), matvec, `emunpu_linear` (GEMM + bias + fused activation) |
| Activations (24) | relu, leaky-relu, relu6, clamp, sigmoid, tanh, gelu (exact + tanh), silu/swish, elu, softplus, hardsigmoid, hardswish, mish, exp, log, sqrt, rsqrt, abs, neg, recip, sign, square, erf |
| Normalization | layer norm, RMS norm (optional affine gamma/beta) |
| Attention / classifier | row-wise softmax, log-softmax (numerically stable) |
| Convolution | conv2d (as im2col + GEMM), im2col, max/avg pool2d |
| Tensor plumbing | transpose, gather (embedding lookup), dtype convert |
| Reductions | sum, max, min, mean, prod, argmax, argmin |
| Elementwise | add, FMA, generic vector ops |
| Control | streams, events, barriers |

Source: `emunpu-sim/README.md`, `include/emunpu/device.h`.

### 1.3 Hardware limits (per instruction, not per call)

| Unit | Limit |
|---|---|
| Matrix engine (MMA) | 16×16×16 per instruction |
| Vector ALU | 256 elements per instruction |
| Special-function unit (activations) | 64 elements per instruction |
| Reduction unit | one row, ≤1024 elements wide |
| DMA | 64 KB per descriptor |

A high-level call (e.g. `emunpu_gemm` on a 1024×1024 matrix) is legal and
tiles itself under the hood. A raw low-level instruction that asks for more
than one of these limits is **rejected**, not silently run. Query the limits
at runtime with `emunpu_device_query_caps()`. Source: `README.md`,
`docs/MICRO_ISA.md`.

### 1.4 Other facts worth knowing

- The core count is a runtime knob for testing:
  `IREE_EMUNPU_NUM_CORES=3 ctest --test-dir build` re-runs the whole test
  suite on 3 cores.
- The public ABI is small and deliberate: every exported symbol is tagged
  `EMUNPU_API` and the library builds with hidden default visibility, so
  only `emunpu_*` symbols are visible (`nm -D --defined-only
  build/libemunpu.so.1 | grep ' T '`).
- There is a live terminal visualizer (`demo/visualizer.py`) and a trace mode
  (`IREE_EMUNPU_TRACE=1`) — useful later for showing Llama-3 tiles moving
  across cores in the presentation.
- Each demo (`mlp_demo`, `cnn_demo`, `transformer_demo`, `lowlevel_demo`)
  checks itself against a double-precision host reference and exits non-zero
  on mismatch. They double as correctness tests (ties to objective BO-5).

---

## 2. `reference/`: the IREE integration example

**Read first:** `emunpu-sim/reference/INTEGRATION.md`.

Everything in `reference/` is a **worked example**, copied verbatim out of a
real IREE backend that used to drive this simulator. It needs IREE to
compile. We do not build it. We read it to see how someone else solved the
"compiler emits a binary, runtime decodes it" problem, because we have to
solve the same problem for our own compiler.

### 2.1 The three-layer picture it describes

```
your compiler backend                 your runtime / HAL driver
──────────────────────                ─────────────────────────
recognizes matmul/add/relu/copy        loads the blob, decodes commands,
tiles matmul into 16x16 blocks ─blob─▶ stages operands into global memory,
emits a command + tile-schedule blob   replays the schedule on the device
                 │                                   │
                 └──────── shared contract ──────────┘
                      reference/emunpu_isa.h ("the ISA")
                                  │
                                  ▼
                      the hardware model you drive
                      include/emunpu/device.h
```

1. **Hardware model** — `include/` + `src/`. What a runtime drives, like a
   CUDA driver: `emunpu_device_create`, `alloc`, `upload`,
   `emunpu_matmul_scheduled` / `emunpu_gemm` / `emunpu_vector_op`, `download`.
2. **The ISA / blob contract** — `reference/emunpu_isa.h` (see §3 below), the
   binary format a code generator emits and a runtime decodes.
3. **The runtime glue** — `emunpu_hw.c` (decoded command → device API call)
   and `emunpu_loader.c` (plugs into IREE's executable-loader interface:
   validates the blob header, dispatches commands).

### 2.2 The one fact that matters most for us

> The blob format only exposes four coarse ops: `MATMUL`, `ADD`, `RELU`,
> `COPY`. Everything else the hardware can do — softmax, norm, conv,
> activations, gather — is reachable only through the high-level device API
> (layer 1), **not** through this blob format. (`emunpu-sim/README.md`,
> "Integrating into your compiler stack", and `docs/ISA.md` Appendix C, which
> sketches what a v5 format would need to add.)

This tells us something important for our own design: if our MLIR pipeline
ever emits a binary blob instead of our current text bytecode, a blob format
modeled on `emunpu_isa.h` would need its own opcode for every op we want
(RMS norm, SiLU, softmax, RoPE, gather, transpose) — it is not something we
get for free just because the simulator supports those ops.

---

## 3. What `emunpu_isa.h` contains

`emunpu_isa.h` is a single C header, dependency-free, included by both a
compiler (C++) and a runtime (C). It is **not** code that runs — it is the
layout of a binary blob, "the ISA" for this integration. Nothing in this
repo currently builds or consumes it (see §4) — it documents a design we can
borrow from, not a dependency we have today.

### 3.1 Blob layout

```
[emunpu_header_t]
[emunpu_export_t   * header.export_count]
[emunpu_command_t  * header.command_count]
[emunpu_matmul_tile_t * header.tile_count]
[string table bytes]   (export names, not NUL-terminated)
```

### 3.2 The pieces

| Struct | Purpose |
|---|---|
| `emunpu_header_t` | Magic number, format version, counts of exports/commands/tiles, string table location. The runtime checks the magic and version before trusting anything else. |
| `emunpu_export_t` | One per compiled function ("dispatch"). Points at its name in the string table and at its slice of the command array (`first_command`, `command_count`). Also records how many bindings and constants it needs. |
| `emunpu_command_t` | One coarse operation: opcode (`MATMUL`/`ADD`/`RELU`/`COPY`), dtype, up to 4 dims, up to 3 operand refs + 1 result ref, plus byte offsets into each operand (for sub-buffer packing). For `MATMUL` it also points into the tile array (`first_tile`, `tile_count`). |
| `emunpu_matmul_tile_t` | One 16×16×16 tile the **compiler** decided at compile time: which core runs it (`core_id`), its M/N/K origin and extents, and flags (`ACCUMULATE` for a running tile, `WRITEBACK` for the last k-tile in a block). |

### 3.3 The key design idea: compile-time tiling

A `MATMUL` command does not say "do a 512×512×512 multiply" and let the
runtime figure out the tiling. The **compiler** decides the full 16×16 tile
grid and which of the 10 cores runs each tile, and bakes that schedule into
the blob as `emunpu_matmul_tile_t` records. The runtime only replays it —
exactly the way a GPU backend bakes in a workgroup grid and the driver just
launches it. This matters for us: it means the compiler, not the runtime, is
responsible for load-balancing matmul work across cores, if we ever adopt
this blob format.

### 3.4 Operand/result encoding

Operands and the result in a command are small integers with this meaning:

- `>= 0` — index into the dispatch's bindings (an input/output/weight buffer).
- `== -1` — unused slot (`IREE_HAL_EMUNPU_REF_UNUSED`).
- `<= -2` — a "scratch" slot: an intermediate buffer the runtime allocates
  for the lifetime of one dispatch, so a fused chain (matmul → add → relu)
  can pass results between commands without a real HAL binding.

### 3.5 Version history baked into the header

- v2 added fused (multi-command) dispatches, scratch-slot refs, and a
  broadcast period on `ADD` (for bias-add style broadcasting, useful for
  `nn.Linear`).
- v4 (current) added the compile-time tile schedule described in §3.3.

---

## 4. Where we are today

Our runtime does **not** use the blob/ISA format from `reference/` at all.
It uses the **high-level device API directly** (layer 1 in the picture
above), called from hand-written C++, not from a compiler-emitted binary.

```
runtime/bytecode-vm/src/main_emunpu.cpp
   → reads bytecode-vm/src/bytecode/*.txt  (our own text bytecode, not emunpu's)
   → VM (vm_core.cpp) dispatches opcodes: ALLOC, COPY_H2D, LAUNCH, COPY_D2H, ...
   → hal/emunpu_device.cpp (emunpuDevice) implements the Device interface by
      calling emunpu_device_create() / emunpu_device_alloc() / ... directly
      (include/emunpu/device.h — the CUDA-like API, not the ISA blob)
```

Concretely, `emunpuDevice::allocate()` in
`runtime/bytecode-vm/src/hal/emunpu_device.cpp:28` calls
`emunpu_device_create` and `emunpu_device_alloc` straight from the device
API. There is no blob, no `emunpu_header_t`, no command table anywhere in
our code today.

| Piece | Status |
|---|---|
| Our own text bytecode format (`.version`, `ALLOC`/`LAUNCH`/...) | Implemented, documented in `doc/bytecode-contract.md` |
| `emunpuDevice` HAL backend, calling the high-level device API | Implemented (memory + dispatch for op_id 0-3) |
| The `emunpu_isa.h` blob format | Not used. Read as reference only. |
| `emunpu_hw.c` / `emunpu_loader.c` glue | Not used (requires IREE; not built) |
| MLIR dialects lowering to anything NPU-shaped | Not started (BO-2, BO-3) |

So the honest answer to "which stage are we at": we have proven that a
hand-written instruction stream can drive `emunpu-sim` correctly end to end
(allocate, upload, matmul-family dispatch, download). We have **not yet**
connected a real compiler output to it — our bytecode today is still
hand-written text files, not something an MLIR lowering pass emits.

---

## 5. What "properly integrate the simulator" means from here

This is the gap between where we are and BO-3/BO-4 (lowering passes +
integration with the emulated NPU backend). Two questions have to be
answered, in roughly this order:

### 5.1 Decide the contract our compiler will emit

Two real options, with the `reference/` files as prior art for option B:

- **Option A — keep growing our own text bytecode**, and have the MLIR
  lowering passes emit *that* instead of driving the device API by hand.
  Lower effort; already partly specified in `doc/bytecode-contract.md`. But
  every new op (softmax, RMS norm, SiLU, RoPE, gather) needs a new `LAUNCH`
  op_id and runtime case, by hand, the same way `op_id` 0-3 were added.
- **Option B — adopt an `emunpu_isa.h`-shaped binary blob.** Closer to how a
  real IREE/LLVM backend would do it (coarse commands + a compile-time tile
  schedule, decoded by a small loader). But as noted in §2.2, the existing
  v4 format only covers `MATMUL`/`ADD`/`RELU`/`COPY` — we would need our own
  v5-style opcodes for softmax/norm/gather/etc. (`docs/ISA.md` Appendix C
  sketches this for the original project; it is not built for us).

This is exactly the kind of decision rule 8 in the project rules asks for:
present the trade-off to the whole team (compiler side + runtime side)
rather than picking silently, since it changes what the compiler side has to
emit.

### 5.2 Build out device coverage for Llama-3 ops

Regardless of which contract wins, the runtime side needs `LAUNCH`
equivalents (op_ids today, or blob opcodes under option B) for the ops
Llama-3 actually needs that we do not dispatch yet: RMS norm, SiLU, softmax,
RoPE, gather (embedding), transpose, elementwise multiply, KV-cache update.
`emunpu-sim` already implements all of the math-level operations (§1.2) —
the gap is only in our dispatch layer (`vm_core.cpp` LAUNCH handling +
`emunpu_device.cpp`), not in the simulator.

### 5.3 Typed, multi-dimension tensors

Today every `ALLOC` on the emunpu backend is hard-coded to fp32, rank 1
(`runtime/CLAUDE.md` §6.5, `emunpu_device.cpp:28`). The simulator itself
already supports 7 dtypes and arbitrary rank (`emunpu_device_alloc` takes
`dtype`, `rank`, `shape[]`) — again, the gap is in our wrapper, not the
simulator. Fixing this is a prerequisite for fp16/bf16/int8 Llama-3 weights.

### 5.4 A real build system

`emunpu-sim` already builds cleanly with `make` or `cmake` and installs a
`find_package`-able target (`emunpu::emunpu`). Our runtime still has no
CMake of its own (`runtime/CLAUDE.md` §6.4) — this is the natural point to
add one, with a backend switch (`USE_EMUNPU` / `USE_CUDA`) and a
`find_package(emunpu)` or a plain `-I`/`-L` pointing at
`emunpu-sim/build`/`emunpu-sim/include`.

---

## 6. Questions and answers

This section answers direct questions raised on 2026-11-16 about the blob,
tiling, data input, and custom operations. Each answer follows the STE rules:
short sentences, active voice, one instruction or fact in each sentence.

### Q1. What exactly are we referencing with "the blob"? How does it differ from our bytecode? Is IREE's `.vmfb` also a blob?

The term "blob" means the binary format in `reference/emunpu_isa.h`. A
compiler writes this binary. The emunpu runtime loader reads this binary. The
binary has five parts in a fixed order: a header, an export table, a command
table, a tile table, and a string table. See §3.1 of this file for the exact
layout.

Our bytecode is different in three ways:

| Property | The blob (`emunpu_isa.h`) | Our bytecode (`doc/bytecode-contract.md`) |
|---|---|---|
| Format | Binary. Fixed-size C structs. | Plain text. One instruction on each line. |
| Who reads it | `emunpu_loader.c`, inside an IREE HAL driver. | Our own VM (`vm_core.cpp`). |
| Tiling | The compiler bakes a 16×16 tile schedule into the blob. | No tile schedule. The simulator's high-level API tiles the matmul for us. |
| Op coverage | 4 ops: `MATMUL`, `ADD`, `RELU`, `COPY`. | 4 ops today: `MATMUL`, `VEC_ADD`, `RELU`, `BIAS_ADD`. More ops need agreement first (`bytecode-contract.md` §5). |

**Yes, IREE's `.vmfb` is also a blob**, in the same general sense: it is a
binary container that an IREE compiler emits and the IREE runtime loads. But
a `.vmfb` is a much bigger container. It holds whole VM bytecode for many
dialects and many backends, with one or more compiled HAL executables inside
it. The `emunpu-cmd-fb` format in `emunpu_isa.h` is only one such inner
executable — the one `.vmfb` would embed for a dispatch that targets the
emunpu backend specifically. Our own bytecode has no relation to `.vmfb`. It
is a separate, simpler format we invented for our own VM.

### Q2. What is the disadvantage of our bytecode compared to the blob?

| Disadvantage | Why it matters |
|---|---|
| Our bytecode is text, not binary. | A real compiled artifact is normally binary. Text is slower to parse and larger on disk for a large Llama-3 program (many weights, many layers). |
| Our bytecode has no compile-time tile schedule. | The compiler cannot control which core runs which tile. The simulator's high-level API decides this for us. We lose compiler-level control over load balancing. |
| Our bytecode has no fused dispatch. | The blob format can fuse several commands into one dispatch and pass results through scratch slots (§3.4). Our bytecode runs one `LAUNCH` at a time, with no equivalent to a scratch slot. |
| Our bytecode has no versioned, formal spec yet. | The blob format has a stable spec (`docs/ISA.md`) with a revision history. Our bytecode contract is still a draft (version 0.2, see `bytecode-contract.md`). |
| Our bytecode needs a hand-written case for every new op. | Each new `LAUNCH op_id` needs a new entry in the contract table and a new case in `vm_core.cpp` and `emunpu_device.cpp`. The blob format has the same problem (§5.2), but its structure already has a place for new coarse ops to slot into. |

The advantage on our side: our bytecode is simple, is already working end to
end, and needs no FlatBuffer library and no IREE dependency to read or write.

### Q3. Does the simulator's high-level API tile for us, or does the compiler have to tile? These seem to contradict.

They do not contradict. They describe two different layers, and which one
applies depends on which API call the caller makes. See §1.1 of this file.

- **Call the high-level device API** (`emunpu_matmul`, `emunpu_gemm`,
  `emunpu_linear`, …): the simulator tiles the matmul for you, at run time,
  inside the library. This is what our runtime does today (§4 of this file).
  The caller gives whole-tensor shapes. The simulator's own scheduler
  (`include/emunpu/scheduler.h`) builds the 16×16 tile grid and assigns tiles
  to cores, at the moment the call runs.

- **Target the blob format** (`emunpu_isa.h`, read by `emunpu_hw.c`): the
  tile schedule is decided once, ahead of time, by the compiler, and is baked
  into the blob as `emunpu_matmul_tile_t` records (§3.3 of this file). At run
  time, the runtime loader only **replays** this fixed schedule. It does not
  compute a new one.

So the simulator can tile for you (high-level API) or can run a schedule you
already decided (blob). It never does both at once for the same call. Our
runtime uses the first path only. The blob path is reference material for a
possible future compiler backend.

### Q4. If the compiler does the tiling, do our opcodes or bytecode need to change?

Only if we move to the blob format (or a blob-like format of our own). Our
bytecode does not need to change for tiling today, because our bytecode does
not use compiler-decided tiling at all — it calls the high-level API, which
tiles internally (Q3).

If we later adopt a blob-style format, three things change:

1. **A `MATMUL` command needs a tile schedule attached.** Each `MATMUL` would
   carry `first_tile` and `tile_count`, pointing into a separate tile table
   (`emunpu_matmul_tile_t[]`), instead of just `M, N, K` scalars.
2. **A new encoding step appears before the runtime ever runs anything.** The
   compiler (not the runtime) must compute the 16×16×16 tile grid and assign
   a `core_id` to each tile, then write that grid into the tile table.
3. **The runtime's job shrinks to replay, not decide.** `vm_core.cpp`/
   `emunpu_device.cpp` would stop calling `emunpu_matmul`/`emunpu_gemm` and
   instead call `emunpu_matmul_scheduled`, handing it the tile array the
   compiler produced.

This is real new work on the compiler side (BO-2/BO-3), not a small flag to
flip. It is the trade-off named in §5.1 of this file (Option A vs Option B).

### Q5. How do we get input data into the model, from the top of the pipeline down to the runtime?

Today, the path runs through the bytecode header and two instructions:

1. **The bytecode file declares input and output slots** in its header, each
   with a slot number, a byte size, and a name (`.input`/`.output` lines,
   `bytecode-contract.md` §2). This is the contract between whoever produces
   the bytecode and the host program that runs it.
2. **The host program fills each input slot** with real data, before it runs
   the VM. In today's demos, this is a hand-written array in `main_emunpu.cpp`.
   For Llama-3, this step becomes: read a weight file (for example, safetensors
   or GGUF) or a tokenized prompt, and copy the bytes into the slot that the
   bytecode's `.input` line names.
3. **`COPY_H2D` moves a slot from host memory to a device buffer.** The VM
   runs this instruction, which calls `emunpuDevice::upload`, which calls
   `emunpu_device_upload` — the real host→device DMA into the simulator's
   global memory.
4. **`LAUNCH` computes on the device buffers.** The result stays on the
   device until copied back.
5. **`COPY_D2H` moves a result back to a host output slot**, which the host
   program then reads with `getOutput`.

For the full pipeline (DSL → MLIR → bytecode → runtime), the missing middle
part is: the compiler must decide, for every weight and every input tensor
in the DSL program, which slot number it gets and emit the matching
`.input`/`ALLOC`/`COPY_H2D` lines. This mapping does not exist yet — it is
part of BO-3 (lowering passes), not something the runtime decides alone.

### Q6. If an operation is not in the simulator, can we write custom logic in the DSL or bytecode, or are we limited to what the simulator supports?

**We are limited to what `emunpu-sim` executes**, because project rule 3 (out
of scope, §2.3) forbids us from changing the simulator. Our DSL and our
bytecode can only ask the simulator to do something it already knows how to
do. We cannot add a new hardware instruction to `emunpu-sim` itself.

Inside that limit, we still have real freedom, because the simulator's op
set is wide (§1.2 of this file):

- **If the simulator's high-level device API already has the operation**
  (any of the 24 activations, RMS norm, softmax, conv2d, gather, transpose,
  reductions, …), we can reach it. The DSL can define the operator, the
  lowering pass can target it, and we add one new `LAUNCH op_id` for it in
  our bytecode (Q2 shows this needs no simulator change, only a runtime-side
  addition).
- **If the operation is a combination of existing ops** (for example, RoPE as
  elementwise multiply and add over rotated slices, or KV-cache update as a
  copy into an offset buffer), the DSL/compiler can express it as a sequence
  of existing `LAUNCH` calls. No new simulator feature is needed — just more
  lowering work.
- **If the operation genuinely needs new hardware behaviour the simulator
  cannot do today** (for example, a new dtype rule, or a new micro-ISA
  instruction), we cannot add it ourselves under the project's own rules.
  The correct move is to ask the supervisor or the Logic Silicon
  co-supervisors whether the simulator can be extended, since that is their
  call, not ours (project rule 7: ask when not sure).

So: custom DSL syntax, yes, as much as we want. Custom bytecode opcodes, yes,
as many as we agree with the compiler team. Custom simulator behaviour, no —
only Logic Silicon or the project's own maintainers can add that, and it is
explicitly out of scope for this FYP.

---

## 7. References

- `emunpu-sim/README.md` — top-level tour, API examples, build instructions.
- `emunpu-sim/reference/INTEGRATION.md` — the three-layer integration picture.
- `emunpu-sim/reference/emunpu_isa.h` — the blob format structs (read-only reference).
- `emunpu-sim/docs/ISA.md` — formal spec of the blob format, byte layout, Appendix C (proposed v5 opcodes).
- `emunpu-sim/docs/MICRO_ISA.md` — the per-core instruction set (DMA/MMA16/vector), §8 maps device-API calls to instructions, §13 is a checklist for adding an opcode.
- `emunpu-sim/docs/MULTICORE.md`, `emunpu-sim/docs/ARCHITECTURE.md` — design deep-dives (file paths inside refer to the original IREE tree, not this repo).
- `runtime/bytecode-vm/hal-design.md` — our own design notes on plugging emunpu behind the VM's `Device` interface.
- `runtime/doc/bytecode-contract.md` — our own text-bytecode format, the thing our compiler must emit today.
- `runtime/PROGRESS.md` — decision log for the runtime side.
