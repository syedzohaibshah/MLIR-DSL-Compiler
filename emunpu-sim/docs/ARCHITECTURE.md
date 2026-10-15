# How the emulated NPU works: ISA, execution, memory, and limits

This document explains the **emulated NPU (`emunpu`)** as a *machine*: what its
instruction set is, how it turns a compiled binary into compute, how its global
memory behaves, and where the emulation departs from a real accelerator.

It complements the other emunpu docs:
- [`ISA.md`](ISA.md) — the executable-format / compiler↔runtime contract.
- [`MICRO_ISA.md`](MICRO_ISA.md) — the normative spec of the instruction set the
  cores execute (every opcode, activation kind, and field encoding).
- [`MULTICORE.md`](MULTICORE.md) — the multi-core simulator deep-dive.
- [`VISUALIZER.md`](VISUALIZER.md) — the live activity visualizer.
- [`../reference/INTEGRATION.md`](../reference/INTEGRATION.md) — wiring the model into a compiler stack.

The two questions this doc answers head-on:
1. **What is the ISA and how does a binary execute?** (§1–§4)
2. **Is its global memory "normal" — load weights once, fetch them at runtime?**
   (§5 — short answer: the device *model* is exactly that; the IREE-compiled
   path currently does not exploit it, and stages operands per op.)

---

## 1. What kind of machine it is

The emulated NPU is a **coarse-grained accelerator**, not a scalar/SIMT
processor. It does not run per-element code the way a CPU or a GPU shader does.
Instead it natively understands **whole-tensor** operations — matmul, elementwise
arithmetic, reductions, activations, softmax, normalization, convolution,
pooling, gather — the way a real NPU/TPU exposes matrix-engine, vector-engine,
and special-function primitives rather than a general instruction stream. (The
*compiled blob* format exposes only the four coarsest of these; see §2a.)

That single design choice is why the backend has *no LLVM codegen*: the compiler
does not lower a matmul to a loop nest of loads/FMAs/stores; it recognizes the
op and emits **one command record** that names the operation, its dimensions,
and its operand buffers. This is the opposite end of the spectrum from the
CUDA/ROCm backends, which are true codegen targets.

Underneath that coarse contract, the *device model* is a realistic multi-core
NPU (`emunpu_arch.h`):

```
              ┌─────────────────────── centralized GLOBAL MEMORY ───────────────────────┐
              │   one contiguous byte arena + descriptor table + coalescing free-list    │
              └───────────▲───────────────────────▲───────────────────────▲─────────────┘
                          │ DMA                    │ DMA                    │ DMA
                  ┌───────┴───────┐        ┌───────┴───────┐        ┌───────┴───────┐
                  │   core 0      │        │   core 1      │  ...   │   core 9      │   (IREE_EMUNPU_NUM_CORES = 10)
                  │ local scratch │        │ local scratch │        │ local scratch │   (256 KB each)
                  │ 16×16 MMA eng │        │ 16×16 MMA eng │        │ 16×16 MMA eng │   (IREE_EMUNPU_MMA_DIM = 16)
                  │ command queue │        │ command queue │        │ command queue │
                  └───────────────┘        └───────────────┘        └───────────────┘
```

Architecture parameters live in `emunpu_arch.h`:
`IREE_EMUNPU_NUM_CORES = 10`, `IREE_EMUNPU_MMA_DIM = 16`,
`IREE_EMUNPU_LOCAL_MEM_BYTES = 256 KB`,
`IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES = 256 MB`.

---

## 2. The ISA — two layers

There are **two** instruction sets, at two levels of abstraction. This split is
deliberate and is the crux of how the machine works.

### 2a. The high-level "executable ISA" — `emunpu_isa.h`

This is the **binary contract** between the compiler and the runtime loader: the
on-disk format of the `emunpu-cmd-fb` blob. It is a small set of plain-C POD
structs (no FlatBuffers) included by *both* halves, so they cannot drift.

A command is one coarse op:

```c
typedef struct iree_hal_emunpu_command_t {
  iree_hal_emunpu_opcode_t opcode;     // MATMUL / ADD / RELU / COPY
  iree_hal_emunpu_dtype_t  dtype;      // F32 / I32
  uint32_t dims[4];                    // MATMUL: {M,N,K,flags}; EW: {count,period}
  uint32_t first_tile, tile_count;     // MATMUL: range into the tile schedule
  int32_t  operand_bindings[3];        // ordinal | scratch-slot | unused
  int32_t  result_binding;
  uint32_t operand_offsets[3];         // byte subspan offset within a binding
  uint32_t result_offset;
} iree_hal_emunpu_command_t;
```

Operands are referenced **indirectly**, never by raw pointer:
- `>= 0` → a **dispatch binding ordinal** (the Nth HAL buffer the runtime hands
  the dispatch). `operand_offsets[]` adds a byte subspan so several tensors can
  share one bound buffer (IREE packs constants this way).
- `<= -2` → a **scratch slot**, a runtime-allocated intermediate that lets a
  fused chain (`matmul → bias-add → relu`) pass results op-to-op without a HAL
  binding.
- `-1` → unused.

The full blob layout (all byte offsets from blob start):

```
[ header        ]   magic 'ENPU', version=4, table counts + offsets
[ exports[]     ]   one per entry point: name + [first_command, command_count)
[ commands[]    ]   the coarse ops above
[ tiles[]       ]   the compile-time matmul tile schedule (see §3)
[ string table  ]   export names (not NUL-terminated)
```

The loader (`emunpu_loader.c`) verifies the header magic/version and that every
table lies within the blob before trusting any offset — a stale v3 blob is
rejected by the version check.

### 2b. The micro-architectural ISA — `emunpu_arch.h`

This is what the **cores actually execute**. The device never runs a coarse
"MATMUL" directly; a scheduler lowers each coarse op into a stream of tiled
*micro-instructions*, one queue per assigned core:

| micro-op        | unit   | meaning                                              |
|-----------------|--------|------------------------------------------------------|
| `DMA_G2L`       | DMA    | copy a (strided 2D) tile global → local scratchpad   |
| `DMA_L2G`       | DMA    | copy a tile local → global                           |
| `MMA16`         | matrix | `C[M×N] += A[M×K]·B[K×N]`, dims ≤ 16 (the tensor core)|
| `MATVEC`        | matrix | `y[M] += A[M×K]·x[K]`                                 |
| `EW`            | vector | elementwise unary/binary (add/sub/mul/div/max/min/pow/relu/copy) |
| `REDUCE`        | vector | row reduction (sum/max/min/mean/prod/argmax/argmin)   |
| `FMA`,`CONVERT` | vector | fused multiply-add; dtype conversion                 |
| `ACT`           | SFU    | 24 parameterized activations (relu…gelu, silu, mish, erf, …) |
| `SOFTMAX`       | SFU    | row-wise softmax / log-softmax (max-subtracted)      |
| `NORM`          | SFU    | row-wise layer / RMS normalization                   |
| `TRANSPOSE`     | vector | `C[N×M] = transpose(A[M×N])`                          |
| `IM2COL`        | vector | convolution patch extraction (conv → GEMM)           |
| `POOL2D`        | vector | 2D max / average pooling                             |
| `GATHER`        | vector | row gather (embedding lookup)                        |

Each `emunpu_instr_t` carries `(space, byte-offset)` operand addresses, element
counts, **byte row-strides** (so a 16×16 sub-block can be DMA'd out of a larger
row-major matrix with no copy), per-op flags, two scalar parameters (an
activation's alpha/beta, a normalization's epsilon), and a window-geometry
sub-struct for the spatial ops. The MMA flags include `ACCUMULATE` (add into C
for the 2nd..nth K-tile) — this is how a >16-wide K is summed across K-tiles.

**Every unit has a width, and an instruction may not exceed it.** The matrix
engine takes at most a 16×16×16 tile; the vector ALU 256 elements per issue; the
special-function unit 64; the reduction/softmax/norm unit one row of at most 1024;
a DMA descriptor 64 KB. An instruction asking for more is rejected
(`emunpu_instr_validate` → `OUT_OF_RANGE`), so **tiling a tensor-sized op into
unit-sized instructions is the producer's job** — the device layer's, a
compiler's, or yours. The limits are queryable at runtime
(`emunpu_device_query_caps`) rather than assumed, and the low-level API
(`emunpu_device_local_alloc` / `_enqueue` / `_launch`, plus
`emunpu_device_set_exec_scope` to choose how many cores an op uses) lets a
code generator drive the cores directly. See
[`MICRO_ISA.md` §2.3](MICRO_ISA.md#23-per-instruction-capability-limits) and
[§10](MICRO_ISA.md#10-programming-the-cores-directly).

The matrix engine, the vector engine, and the special-function unit are the same
three-block structure a real NPU has: a convolution, for instance, is not a
dedicated engine — `IM2COL` marshals the patches and the ordinary 16×16 tensor
cores do the arithmetic. [`MICRO_ISA.md`](MICRO_ISA.md) is the normative
per-opcode specification (fields, semantics, error conditions, and the exact
activation formulas).

The engine is **dtype-generic**: every kernel reads/writes through
`emunpu_dtype_load/store` (in `emunpu_dtype.c`), which widen fp16/bf16/integers
exactly, so the 7 supported dtypes and mixed precision (e.g. fp16 in / fp32
accumulate) fall out for free. (The *IREE HAL path* currently only plumbs
f32/i32 through `emunpu_isa.h`; the wider set is reachable via the device-native
API.)

---

## 3. Tiling is a compile-time decision (the CUDA/ROCm analog)

A matmul larger than 16×16 must be decomposed into 16×16 blocks and spread over
the 10 cores. **That decision is made by the compiler**, not the runtime —
matching how CUDA/ROCm bake the workgroup grid into the HAL executable.

- The compiler (`EmuNPUTarget.cpp::buildMatmulTileSchedule`) computes the tile
  grid `ceil(M/16) × ceil(N/16) × ceil(K/16)` and assigns each output block to a
  core round-robin (`core_id = outputTileIndex % NUM_CORES`). It emits one
  `iree_hal_emunpu_matmul_tile_t` per (output-block × K-tile) into the blob's
  tile section; the MATMUL command points at the range with `first_tile` /
  `tile_count`.
- A tile record carries only **shape-relative coordinates**
  (`m0,n0,k0, tm,tn,tk`) + `core_id` + flags (`ACCUMULATE`/`WRITEBACK`) — never
  absolute memory offsets, because those depend on runtime buffer staging.
- The runtime **replays** the schedule (`emunpu_run_gemm_schedule`): for each
  tile it adds the operand base to the shape coordinates and issues the
  `DMA_G2L → MMA16 → DMA_L2G` micro-instruction stream to the assigned core —
  exactly as a CUDA kernel computes per-block addresses from `blockIdx`. It
  makes **no tiling decision**.

Elementwise ops (add/relu/copy) are *not* compile-time tiled: they are an
embarrassingly-parallel 1D element range that the runtime simply splits across
cores. There is no block-shape choice to hoist, so this is left in the runtime
by design (documented, not an oversight).

> The device-native API (`emunpu_schedule_gemm`, used by `emunpu_device_test`
> and the Python device demos) keeps its own on-device tiler — like a library
> (cuBLAS) tiling internally. Only the IREE compiler→HAL path moved the decision
> to compile time. Both paths emit the same per-core micro-instruction streams.

---

## 4. How a binary executes, end to end

```
iree-compile  ──►  .vmfb  (contains the emunpu-cmd-fb blob per dispatch)
                     │
iree-run-module ─────┤  VM bytecode issues hal.command_buffer.dispatch
                     ▼
  emunpu_loader.c  : iree_hal_emunpu_executable_create
        - verify_blob (magic/version/bounds)
        - copy blob, parse header → exports/commands/tiles/string-table views
                     │  dispatch(ordinal)
                     ▼
  emunpu_loader.c  : issue_call(ordinal, dispatch_state)
        - the dispatch records a SINGLE workgroup; (id != 0) is a no-op guard
        - size & allocate scratch slots for fused intermediates
        - build merged [bindings | scratch] pointer/length arrays
        - for each command in the export's range:
              (optional IREE_EMUNPU_TRACE log)
              remap scratch refs → merged ordinals
              iree_hal_emunpu_hw_execute_command(cmd, tiles, ptrs, lengths)
                     │
                     ▼
  emunpu_hw.c      : resolve each operand ref+offset → host pointer (bounds-checked)
        MATMUL  → decode_tiles (validate every tile vs M/N/K)
                → multicore_matmul: stage into device global mem,
                  emunpu_matmul_scheduled → emunpu_run_gemm_schedule (replay)
        ADD/RELU/COPY → multicore_elementwise: range-split across cores
                     │
                     ▼
  emunpu_device.c / emunpu_core.c : cores drain their queues
        DMA_G2L → MMA16 (×K-tiles, accumulate) → DMA_L2G  (per tile)
```

Two important properties of the loader:
- **One workgroup per dispatch.** The coarse op runs exactly once over the whole
  tensor; `export_info` reports `workgroup_size = {1,1,1}` and `issue_call`
  early-returns for any non-zero workgroup id, so an accidental larger grid can
  never re-run and corrupt the result.
- **No native function pointers.** `dispatch_ptrs = NULL`; dispatches are
  *interpreted* in `issue_call` by walking the command list — there is no
  machine code to call.

`IREE_EMUNPU_TRACE=1` logs each coarse op as the engine executes it (with a
running counter and binding/scratch refs) — the proof that a workload truly ran
on the NPU and which ops did vs. fell to the host.

---

## 5. Global memory — is it "normal" global memory?

This is the question worth being precise about, because the answer differs
between the **device model** and the **IREE-compiled path**.

### 5a. The device model: yes — a real, persistent global memory

`emunpu_global_mem.{h,c}` implements exactly what you'd expect of an
accelerator's HBM/global memory:

- **One contiguous byte arena** (default 256 MB), the single address space all
  cores DMA in and out of. Cores never allocate at runtime — they only DMA into
  regions the device handed them.
- A **descriptor table**: each live allocation has an id, dtype, shape, byte
  offset (its "global address"), size, and debug name. This table *is* the
  authoritative memory layout, and can be serialized to / parsed from a
  FlatBuffer (`emunpu_mem.fbs`) so the layout is inspectable and portable.
- A **coalescing free-list bump allocator**: freed regions are reused and merged;
  otherwise a watermark bumps.

Through the **device-native runtime API** (`emunpu_device.c`, the CUDA-style
`alloc / upload / matmul / download / free` surface), this behaves precisely
like normal global memory:

> **You allocate a tensor once, `upload` weights into it, and they stay resident
> at a fixed global address. Every subsequent compute op re-fetches (re-DMAs)
> them from that address — across as many matmuls as you like — until you
> explicitly `free` it.**

So *yes*: in the device model, weights loaded once can be fetched repeatedly at
runtime, exactly like a real NPU keeping weights resident in HBM and streaming
them into the MMA engine each layer. `emunpu_device_test.c` and the Python
device demos exercise this resident-weights model directly.

### 5b. The IREE-compiled path: the buffers are IREE's, staged per op

When you run a `.vmfb` through `iree-run-module --device=emunpu`, memory
ownership flips. **IREE** owns the tensors (weights, activations, results) as HAL
buffers in host memory, and manages their lifetimes. The emunpu engine does not
keep them in its arena. For each coarse op (`emunpu_hw.c`), the engine:

1. `alloc`s lhs/rhs/out in the device arena,
2. `upload`s the operand bytes from the HAL binding pointers,
3. computes (replaying the tile schedule across the cores),
4. `download`s the result back to the HAL output buffer,
5. **`free`s all three** (in reverse, so the free-list coalesces back to empty).

The reason this works at all — and the reason it's acceptable — is that the
emulated device computes directly on **host/unified memory**: there is no
discrete device DRAM to cross. The hetero topology even asserts
`unified_memory = true` for this reason. Staging into the arena is mostly to
reuse the multi-core scheduler and to sidestep IREE's input/output buffer
aliasing (inputs are fully staged before any output byte is written).

**Consequence:** on the IREE path, weights are **re-uploaded and freed on every
dispatch** rather than held resident across invocations. That is a
*simplification of the emulation*, not a property of real hardware — a real NPU
(or a more faithful model) would load weights into global memory once and
re-fetch the resident copy each inference. Making the HAL path keep weights
resident (e.g. caching constant bindings in the arena keyed by buffer identity)
is a natural future extension; today it does not.

### Summary table

| Aspect | Device-native API (`emunpu_device.c`) | IREE-compiled path (`emunpu_hw.c`) |
|---|---|---|
| Who owns tensors | the emunpu device (in its arena) | IREE HAL buffers (host memory) |
| Weights resident across calls | **Yes** — alloc once, fetched each op | **No** — uploaded/freed per dispatch |
| Global address space | persistent arena + descriptor table | arena used only as per-op scratch |
| Matches "normal" global memory | **Yes** | Partially — unified memory, not resident |
| Used by | `emunpu_device_test`, Python device demos | `iree-run-module`, `.vmfb` models |

---

## 6. Limitations compared to real hardware

The emulation is faithful to the *structure* of an NPU but intentionally omits
the physics. Concrete gaps:

- **No timing / performance model.** Execution is functionally synchronous and
  single-threaded; cores "fan out" logically but are drained sequentially. There
  is no cycle count, no latency, no bandwidth limit, no contention — results are
  bit-exact but tell you nothing about *speed*. Real NPUs are dominated by
  memory bandwidth and overlap.
- **No real DMA cost or overlap.** `DMA_G2L`/`DMA_L2G` are immediate `memcpy`s.
  Real hardware overlaps DMA with compute (double-buffering) and is bottlenecked
  by the global-memory↔local-scratchpad link.
- **Scratchpad capacity is enforced, bank pressure is not.** A local allocation
  past 256 KB fails, and every instruction is checked against its unit's width,
  but there is no model of register/bank conflicts, port limits, or the number of
  instructions a unit can have in flight.
- **The compiled-blob op set is narrower than the hardware.** The device itself
  now has a realistic vector/special-function engine (activations, softmax,
  layer/RMS norm, transpose, im2col, pooling, gather — see
  [`MICRO_ISA.md`](MICRO_ISA.md)), all reachable through the device/runtime API.
  The coarse `emunpu-cmd-fb` blob, however, is still v4 and can only name
  matmul/add/relu/copy, so a *compiled* model still offloads anything else to the
  host (see the heterogeneous docs). Closing that gap is a format change, not a
  hardware one: [`ISA.md` Appendix C](ISA.md#appendix-c--non-normative-proposed-v5-opcodes).
- **Weights are not resident on the IREE path** (§5b) — no on-device weight
  caching, no KV-cache-style persistent state across dispatches.
- **No concurrency / multi-stream timing.** Streams and events exist in the
  device API for ordering, but there is no asynchronous engine — a real NPU
  overlaps queues, DMA, and compute.
- **Unified host memory.** The model computes on host buffers; there is no
  discrete device memory, no PCIe/transfer cost, and the hetero path elides
  cross-device copies it asserts are free. A discrete NPU would need genuine
  transfers.
- **Single fixed configuration baked into headers.** Core count, MMA dim, and
  memory sizes are compile-time `#define`s shared by both halves; they are not
  queried/negotiated at runtime the way a real driver discovers device caps.

These omissions are the point: `emunpu` exists to demonstrate the **compiler↔
runtime contract** (recognize coarse ops → emit a command+tile blob → load →
replay across cores), and to be a clean, dependency-free, bit-exact reference —
not to model performance.
