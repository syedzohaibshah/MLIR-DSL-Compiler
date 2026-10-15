# Multi-core NPU architecture (`emunpu` simulator)

This extends the original coarse-grained `emunpu` backend (see
[`README.md`](README.md)) into a **realistic multi-core NPU simulator**: a
centralized global memory subsystem feeding an array of independent compute
cores, each with its own local scratchpad, a tensor-core-style MMA engine, a
private command queue, and DMA access to global memory. A scheduler tiles
workloads into 16×16 blocks and distributes them across the cores.

The simulator is **IREE-independent** (plain C, only the standard library), so
it builds and unit-tests standalone and is wrapped by the IREE HAL driver. It is
the new "hardware model" the README's §13 anticipated. Session 1 routed compute
through the multi-core engine with no compiler/ISA change; **session 2 (§9)
extended the `emunpu-cmd-fb` blob to v4** so the compiler ships a matmul tile
schedule the runtime replays, instead of the runtime deciding the tiling.

> Status: **session 1** built the runtime simulator + device API + HAL
> integration (on-device tiling). **Session 2 moved the matmul tiling *decision*
> into the compiler**: the `emunpu-cmd-fb` blob now carries a per-matmul tile
> schedule (16×16 block grid + per-core assignment, see §9), and the runtime
> *replays* it instead of deciding it — matching how IREE's CUDA/ROCm backends
> bake the workgroup grid at compile time. The device-native API path still tiles
> on-device; both share the same per-core instruction emission.

---

## 1. Architecture overview

```
                         host
                          │  device API (alloc/upload/matmul/…/streams/events)
                          ▼
        ┌─────────────────────────────────────────────────────┐
        │                emunpu_device_t                        │
        │                                                       │
        │   ┌───────────────── Global Memory ────────────────┐ │
        │   │  one byte arena + coalescing allocator          │ │
        │   │  descriptor table  (FlatBuffer layout)          │ │
        │   └───────────▲───────────────────────▲─────────────┘ │
        │     DMA g2l/l2g │                       │ DMA           │
        │   ┌────────────┴───┐   ┌────────────┐  ┌┴───────────┐  │
        │   │   core 0       │   │   core 1   │… │   core 9   │  │
        │   │ local scratch  │   │ local mem  │  │ local mem  │  │
        │   │ MMA16 engine   │   │ MMA16      │  │ MMA16      │  │
        │   │ command queue  │   │ queue      │  │ queue      │  │
        │   └────────────────┘   └────────────┘  └────────────┘  │
        │            ▲ scheduler fans 16×16 tiles across cores    │
        └─────────────────────────────────────────────────────┘
```

| Parameter | Value | Where |
|-----------|-------|-------|
| Compute cores | **10** | `IREE_EMUNPU_NUM_CORES` |
| MMA tile | **16×16** | `IREE_EMUNPU_MMA_DIM` |
| Local scratchpad / core | 256 KB | `IREE_EMUNPU_LOCAL_MEM_BYTES` |
| Global memory (default) | 256 MB | `IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES` |

> **How the memory hierarchy and cores actually work — see [§10](#10-how-the-cores-and-global-memory-work-deep-dive)**
> for the two-level memory model (shared global arena + per-core scratchpad), the
> allocator, the DMA-in/compute/DMA-out tile lifecycle, and the execution model.

---

## 2. File map (runtime — `runtime/src/iree/hal/drivers/emunpu/`)

| File | Role |
|------|------|
| `emunpu_arch.h` | Architecture params, status codes, dtype enum, and the **micro-ISA** (per-core instructions: DMA / MMA16 / MATVEC / EW / REDUCE / FMA / CONVERT / ACT / SOFTMAX / NORM / TRANSPOSE / IM2COL / POOL2D / GATHER — spec in [`MICRO_ISA.md`](MICRO_ISA.md)). IREE-free. |
| `emunpu_ops.c` | The sub-operation name tables (opcode / elementwise / reduction / activation / norm / pool) and the activation unit's scalar transfer function `emunpu_act_apply`. One table per enum, shared by the kernels, the trace, and the visualizer. |
| `emunpu_dtype.c` | Element sizes/names, **fp16/bf16 ↔ fp32** conversion, and the unified load/store-as-`double` accessors that make every kernel dtype-agnostic (so mixed precision + datatype conversion are free). |
| `emunpu_global_mem.{h,c}` | **Global memory subsystem**: one byte arena + a coalescing free-list allocator + a tensor descriptor table. |
| `emunpu_mem.fbs` + `emunpu_mem_fb.{h,c}` | **FlatBuffer layout** of the global memory descriptor table (serialize/parse via flatcc). |
| `emunpu_core.{h,c}` | A single **compute core**: local scratchpad, command queue, and the tensor engine that executes the micro-ISA. |
| `emunpu_scheduler.{h,c}` | **Tile scheduler**: emits the per-core DMA+MMA16 streams for 16×16 GEMM tiles. Two entry points share the same per-tile emission — `emunpu_schedule_gemm` *decides* the tiling on-device (device-native API), `emunpu_run_gemm_schedule` *replays* a compiler-decided schedule (IREE HAL path). |
| `emunpu_device.{h,c}` | **Device + runtime API** (below). Owns global memory + the 10 cores + streams. |
| `emunpu_device_test.c` | Standalone functional test (registered as ctest `emunpu_device_test`). |
| `emunpu_hw.c` | The HAL bridge: decodes a command, **replays the matmul's compile-time tile schedule** across the cores (`iree_hal_emunpu_hw_decode_tiles` → `iree_hal_emunpu_hw_multicore_matmul`). |

---

## 3. Datatypes (`emunpu_dtype_t`)

`FP32`, `FP16`, `BF16`, `INT32`, `INT16`, `INT8`, `UINT8`.

Every kernel reads operands through `emunpu_dtype_load(... ) -> double` and writes
through `emunpu_dtype_store(...)` (round-to-nearest-even for fp16/bf16; rounding +
saturation for integers). Consequences:

* **Mixed precision**: the MMA reads `a_dtype`/`b_dtype` inputs and writes a
  (typically wider) `c_dtype` accumulator — e.g. `fp16×fp16→fp32` or `i8×i8→i32`,
  exactly the tensor-core accumulate model.
* **Datatype conversion**: the `CONVERT` op is just load-as-A, store-as-C.

---

## 4. Micro-ISA (per-core instructions)

A core executes a queue of `emunpu_instr_t`. Operands are `(memory space, byte
offset)` pairs into either global memory or the core's local scratchpad, with
byte row-strides so a sub-tile of a row-major matrix is DMA'd without a copy.

| Opcode | Meaning |
|--------|---------|
| `DMA_G2L` / `DMA_L2G` | 2-D (strided) tile copy global↔local |
| `MMA16` | `C[M×N] (+)= A[M×K]·B[K×N]`, dims ≤ 16, std or transpose-B, fp32/i32 accumulate |
| `MATVEC` | `y[M] (+)= A[M×K]·x[K]` |
| `EW` | elementwise add/sub/mul/div/max/min/pow/relu/copy (+ broadcast period) |
| `REDUCE` | row reduction sum/max/min/mean/prod/argmax/argmin |
| `FMA` | `c[i] = a[i]·b[i] + c[i]` |
| `CONVERT` | `c[i] = convert(a[i])` across dtypes |
| `ACT` | 24 parameterized activations (relu/leaky/relu6/clamp/sigmoid/tanh/gelu/silu/elu/softplus/hardswish/mish/exp/log/sqrt/rsqrt/…) |
| `SOFTMAX` | row-wise softmax / log-softmax, max-subtracted |
| `NORM` | row-wise layer / RMS normalization (epsilon parameter) |
| `TRANSPOSE` | `C[N×M] = transpose(A[M×N])` |
| `IM2COL` | convolution patch extraction, so conv2d becomes a GEMM |
| `POOL2D` | 2D max / average pooling |
| `GATHER` | `c[i,:] = a[idx[i],:]` (embedding lookup) |

Each opcode is bounded by the width of the unit that issues it (16×16×16 for the
matrix engine, 256 elements for the vector ALU, 64 for the special-function unit,
one ≤1024-wide row for the row unit, 64 KB for a DMA descriptor); an instruction
exceeding its limit is rejected, and tiling to those limits is the producer's job.
[`MICRO_ISA.md`](MICRO_ISA.md) is the normative per-opcode specification (fields,
exact semantics, capability limits, activation formulas, error conditions) and
documents the low-level API for issuing instructions yourself.

### Tiled GEMM (what the schedule drives)

For each 16×16 output tile (assigned round-robin to a core), over each K-tile:
`DMA A-subtile g→l ; DMA B-subtile g→l ; MMA16 (acc= on first K-tile, acc+= after)`
then `DMA C-tile l→g`. On the HAL path this sequence is driven by the compiler's
tile schedule (see §9); the on-device scheduler produces the identical streams.
Example trace (`IREE_EMUNPU_TRACE=1`) for a 40×48×24 GEMM:

```
[npu] GEMM 40x48x24 -> 9 compiler-scheduled tiles across 10 cores
[core0] DMA.g2l fp32 rows=16 cols=16 (...)
[core0] DMA.g2l fp32 rows=16 cols=16 (...)
[core0] MMA16 fp32*fp32->fp32 M=16 N=16 K=16 acc=
[core0] DMA.g2l fp32 rows=16 cols=8 (...)
[core0] DMA.g2l fp32 rows=8 cols=16 (...)
[core0] MMA16 fp32*fp32->fp32 M=16 N=16 K=8 acc+=
[core0] DMA.l2g fp32 rows=16 cols=16 (...)
[core1] ...
```

---

## 5. Device / runtime API (`emunpu_device.h`)

CUDA/HIP-runtime-flavored surface over the device:

* **Memory**: `emunpu_device_alloc` / `_free`, `emunpu_device_upload` / `_download`.
* **Compute** (each runs immediately on the default stream, or is recorded on a
  given stream): `emunpu_matmul`, `emunpu_gemm` (+bias +relu epilogue),
  `emunpu_matvec`, `emunpu_vector_op`, `emunpu_reduce`, `emunpu_fma`,
  `emunpu_convert`. GEMM is tiled across the cores; vector/reduction/fma/convert
  split their element range across the cores.
* **ML operations**: `emunpu_activation`, `emunpu_linear` (GEMM + bias + any
  fused activation), `emunpu_softmax`, `emunpu_layernorm`, `emunpu_rmsnorm`,
  `emunpu_transpose`, `emunpu_conv2d`, `emunpu_im2col`, `emunpu_pool2d`,
  `emunpu_gather`. Each is split along the axis that keeps a core's slice
  self-contained — lane-sized element ranges, single rows, or plane row bands
  ([`MICRO_ISA.md` §9](MICRO_ISA.md#9-multi-core-partitioning)). How many cores
  take part is itself selectable: `emunpu_device_set_exec_scope`. `conv2d` is a
  composite: it lowers to `im2col` + a transpose-B GEMM on the same tensor cores,
  with bias/activation epilogues and a per-image transpose back to NCHW,
  allocating and releasing its own scratch.
* **Hardware capabilities + low-level control**: `emunpu_device_query_caps`
  (core count, memory sizes, and the per-instruction width of every functional
  unit), `emunpu_device_set_exec_scope` (how many cores the next operations use),
  and the raw programming surface `emunpu_device_local_alloc` /
  `emunpu_device_enqueue` / `emunpu_device_launch` / `emunpu_device_core_stats`
  for issuing instruction streams yourself.
* **Streams / events / sync**: `emunpu_stream_create` / `_synchronize`,
  `emunpu_event_create` / `_record` / `_synchronize`, `emunpu_stream_wait_event`,
  `emunpu_device_synchronize`, `emunpu_device_barrier`. Execution is functionally
  synchronous + single-threaded: streams record ordered command lists drained on
  synchronize (no concurrency/timing model — see fidelity choice below).

---

## 6. Build & test

The simulator is part of the `iree_hal_drivers_emunpu_emunpu` library and builds
with the normal `IREE_HAL_DRIVER_EMU_NPU=ON` flag (already in `build.sh`).

```bash
# Build the driver + tools + the standalone simulator test.
cmake --build ../iree-build --target iree-run-module \
      iree_hal_drivers_emunpu_emunpu_device_test

# Functional unit test (GEMM tiling/ragged/multi-core, all 7 dtypes, mixed
# precision, vector/reduce/fma/convert, streams/events/barrier, GMEM reuse):
cd ../iree-build && ctest -R emunpu_device_test --output-on-failure

# It also compiles fully standalone (no IREE), for fast iteration:
clang -std=c11 -I runtime/src -lm \
  runtime/src/iree/hal/drivers/emunpu/emunpu_{dtype,global_mem,core,scheduler,device}.c \
  runtime/src/iree/hal/drivers/emunpu/emunpu_device_test.c -o /tmp/emunpu_test && /tmp/emunpu_test
```

End-to-end through IREE (the matmul tiles across the cores):

```bash
IREE=../iree-build/tools
$IREE/iree-compile mm.mlir --iree-hal-target-backends=emunpu -o mm.vmfb   # 40x24 * 24x48
IREE_EMUNPU_TRACE=1 $IREE/iree-run-module --device=emunpu --module=mm.vmfb \
  --function=matmul --input="40x24xf32=1.0" --input="24x48xf32=1.0"       # -> 24.0 everywhere
# PyTorch MLP (transpose-B matmuls routed multi-core), exact vs torch:
IREE_EMUNPU_TRACE=1 python emunpu_demos/emunpu_pytorch.py
```

---

## 7. Fidelity & scope

* **Functional + architectural**, not cycle-accurate. Results are bit-exact;
  the architecture (per-core local memory, real DMA copies, per-core queues,
  16×16 tile scheduling) is modeled and observable in the trace. There is no
  latency/throughput/contention model.
* **HAL path** uses `fp32`/`i32` (the coarse ISA's dtypes); the full 7-dtype +
  mixed-precision support is a device-API capability exercised by the standalone
  test. Both matmul **and** elementwise (add/relu/copy) route through the
  multi-core device: matmul tiles across cores, and elementwise splits its
  element range across cores. A *broadcast* add (the `nn.Linear` row bias) also
  splits, with the per-core chunk rounded up to a multiple of the broadcast
  period so every slice starts on a period boundary and `i mod period` stays
  computable locally.

## 8. Live visualizer (`emunpu_demos/emunpu_viz.py`)

A dependency-free terminal dashboard animates the cores while a workload runs.
The engine writes a line-oriented event stream (run start, GEMM tiling, every
per-core instruction, global-memory usage, barriers) whenever the environment
variable `IREE_EMUNPU_VIZ` names a file (`emunpu_viz.{h,c}`); the Python tool
tails it and renders. Emission is a no-op when the variable is unset.

```bash
# Terminal 1 — start the visualizer (creates/owns the event file):
python emunpu_demos/emunpu_viz.py /tmp/emunpu.events            # --delay tunes animation speed

# Terminal 2 — run any emunpu workload pointing IREE_EMUNPU_VIZ at the same file:
IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_demos/emunpu_pytorch.py
IREE_EMUNPU_VIZ=/tmp/emunpu.events ../iree-build/tools/iree-run-module \
  --device=emunpu --module=mm.vmfb --function=matmul \
  --input="96x96xf32=1.0" --input="96x96xf32=1.0"
```

The dashboard shows a global-memory bar, the current GEMM + its tile/core fan-out,
and one row per core with its last op, a relative op-load bar, MMA/DMA/vector
counts, and a recent-activity pulse:

```
 Global memory  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  108.0 KB / 2.2 MB
 Current GEMM   96x96x96  -> 36 tiles / 10 cores
 core0  │ DMA.l2g  │ ████████████████████████ │ mma 24    dma 52    vec 0    ●
 core6  │ DMA.l2g  │ ██████████████████░░░░░░ │ mma 18    dma 39    vec 0    ●
 ...
```

`python emunpu_demos/emunpu_viz.py <file> --replay` replays an existing event file once (handy
for inspecting a captured run). Run the visualizer *first* so it owns the file,
then launch inference in the other terminal.

Full write-up of the event stream, producer/consumer split, schema, and how to
read the dashboard: [`VISUALIZER.md`](VISUALIZER.md).

## 9. Compiler-side matmul tiling (shipped)

The matmul **tiling decision** is now made at compile time, mirroring IREE's
CUDA/ROCm backends (where `KernelConfig` picks tile sizes and the workgroup grid
is baked into the HAL executable; the runtime driver only launches it). For
emunpu:

- **Contract (`emunpu_isa.h`, blob v4):** the blob gains a tile section,
  `header | exports | commands | tiles | string-table`. Each MATMUL command
  points at `tiles[first_tile .. first_tile+tile_count)`. A tile record
  (`iree_hal_emunpu_matmul_tile_t`) is `{core_id, m0,n0,k0, tm,tn,tk, flags}` —
  shape-relative coordinates only, never absolute offsets (those depend on
  runtime buffer staging), with `ACCUMULATE`/`WRITEBACK` flags.
- **Compiler (`EmuNPUTarget.cpp::buildMatmulTileSchedule`):** tiles the M/N/K
  space into `IREE_EMUNPU_MMA_DIM` blocks and round-robins each output tile
  across `IREE_EMUNPU_NUM_CORES` — the decision that used to live in
  `emunpu_schedule_gemm`.
- **Runtime:** the loader parses the tile section; `emunpu_hw.c` decodes/validates
  it and calls `emunpu_run_gemm_schedule`, which *replays* the schedule with no
  tiling decision (the analog of a GPU kernel computing per-block addresses from
  `blockIdx` at launch). `IREE_EMUNPU_MMA_DIM`/`IREE_EMUNPU_NUM_CORES` are the
  physical device params both halves agree on.

**Not yet moved (future work):** elementwise (add/relu/copy) still splits its
element range in the runtime — a 1-D embarrassingly-parallel split with no block
shape to decide. Global/local **memory planning** and full **DMA/micro-ISA
emission** (shipping the entire per-core instruction stream of `emunpu_arch.h` in
the blob, reducing the runtime to a pure executor) also remain on-device; the
tile schedule is the first, highest-value slice of that.

---

## 10. How the cores and global memory work (deep dive)

The device is a classic accelerator memory split: **one big shared global memory**
that every buffer is visible in, and **per-core private scratchpads** that are
small and fast. Cores never compute on full-size global buffers — they **DMA tiles
in, compute locally, and DMA results back out**, exactly the GPU shared-memory /
NPU-SRAM pattern.

```
Global Memory (256 MB shared arena)  ── DMA g2l / l2g ──▶  Core local scratchpad (256 KB)
        owned by the device                                       owned by one core
```

### 10.1 Global memory subsystem (`emunpu_global_mem.c`)

Global memory is **one contiguous `malloc`'d byte arena** (`gm->arena`, default
256 MB; `emunpu_global_mem.c:25`) plus two bookkeeping structures:

* **A coalescing free-list allocator.** `emunpu_arena_carve` (`:45`) does *first-fit*
  over a free-block list and only bumps a watermark when nothing fits (`:63`).
  `emunpu_arena_release` (`:73`) returns a region and **merges adjacent free
  blocks**; if the freed region touches the watermark it simply lowers the
  watermark (`:94`). Every allocation is 64-byte aligned (`:14`) so any 16×16 tile
  stays naturally aligned. If the free list overflows, the region leaks until
  reset — "bounded and benign for an emulator" (`:103`).
* **A descriptor table** (`descs[]`, `emunpu_global_mem.h:61`). Each live
  allocation is an `emunpu_tensor_desc_t` — `{id, dtype, rank, shape, offset,
  size, name}` (`emunpu_global_mem.h:36`). This table is the device's
  authoritative memory layout and is serializable to a FlatBuffer (`emunpu_mem.fbs`).

The key abstraction: a tensor's **global "address" is just its byte offset into the
arena** (`emunpu_global_mem_offset`, `:189`); cores address global memory by raw
offset, resolved to a host pointer via `emunpu_global_mem_at` (`:183`). Cores never
allocate here at runtime — the device hands them regions to DMA against.

### 10.2 A compute core (`emunpu_core.c`)

Each of the 10 cores owns three things (`emunpu_core.h:27`):

1. **Local scratchpad** — a 256 KB buffer (`:24`) with a dead-simple **bump
   allocator**: `emunpu_core_local_alloc` aligns to 64B and bumps a watermark
   (`:44`); `emunpu_core_local_reset` frees every local tile at once by zeroing the
   watermark (`:54`). One dispatch's tiles live here, then it's reset.
2. **A private command queue** — a growable array of `emunpu_instr_t` (`:56`,
   doubles on demand). The scheduler fills it; the core drains it.
3. **The execution engine** — `emunpu_core_run` walks the queue **in order,
   single-threaded**, executing each instruction and then clearing the queue (`:504`).

**Operand resolution** is the heart of the two-space model: `emunpu_core_resolve`
(`:79`) takes a `(space, byte_offset)` pair and bounds-checks it against either the
core's local capacity (`SPACE_LOCAL`) or the global arena capacity
(`SPACE_GLOBAL`), returning a host pointer. Every kernel resolves its A/B/C
operands through this, so the same instruction works whether operands live local or
global. The **DMA** instruction (`emunpu_core_dma`, `:188`) is a 2-D strided
`memcpy` between the two spaces — `DMA_G2L` reads global / writes local, `DMA_L2G`
the reverse; byte row-strides pull a sub-tile out of a larger row-major matrix with
no repacking (`:196`).

### 10.3 How they work together: a tiled GEMM

The scheduler (`emunpu_scheduler.h`) fans work across cores. For a GEMM it tiles the
M/N/K space into 16×16 blocks and **round-robins output tiles across the 10 cores**.
For each output tile, on its assigned core, it enqueues:

```
for each K-tile:
    DMA_G2L  A-subtile  global → local
    DMA_G2L  B-subtile  global → local
    MMA16    C_local (+)= A_local · B_local     (acc= on first K-tile, acc+= after)
DMA_L2G  C-tile  local → global
```

The MMA engine (`emunpu_core_mma16`, `:224`) enforces the ≤16 dimension (`:228`),
reads inputs in their own dtypes, and accumulates in a wider precision (`double`
internally, modeling fp32/i32 accumulate) — this is where mixed precision comes
from. The `ACCUMULATE` flag selects overwrite-vs-add across K-tiles.

Two entry points share that per-tile emission:

* **`emunpu_schedule_gemm`** — the device-native API path *decides* the tiling
  on-device.
* **`emunpu_run_gemm_schedule`** — the **IREE HAL path** *replays* a tile schedule
  the **compiler already decided** (`EmuNPUTarget.cpp::buildMatmulTileSchedule`, see
  [§9](#9-compiler-side-matmul-tiling-shipped)), so for compiled programs the
  runtime does no tiling — it just executes the per-core streams.

### 10.4 Execution model (important caveat)

It is **functionally accurate, not cycle-accurate**. "Fan-out across cores" means
each core's queue is filled and then `emunpu_core_run` is called on each —
sequentially, single-threaded. Results are bit-exact and the architecture (per-core
local memory, real DMA copies, per-core queues, 16×16 tiling, round-robin
assignment) is real and observable in the `IREE_EMUNPU_TRACE=1` trace, but there is
**no latency / throughput / contention model**. The parallelism is structural, not
timed.

---

## 11. How reductions map to cores (no cross-core dependency)

A natural question: if each core computes independently, how does a **reduction**
work — doesn't that need cores to combine partial results? Two facts make it work
without any cross-core coordination.

**Global memory is shared.** The cores are not fully isolated: only the 256 KB
local scratchpad is private. The 256 MB **global arena is shared by all cores**
(§10.1), reachable by DMA, so cores *could* communicate through it. They mostly
don't need to — because of how the work is partitioned.

**The reduction is partitioned along independent rows.** The `REDUCE` op is a
**row reduction**: each of `M` rows of width `N` reduces independently to one output
element (`emunpu_core_reduce`, `emunpu_core.c:379`). The device splits the work by
assigning **whole rows to cores** — never by splitting a single row's reduction
across cores (`emunpu_exec_reduce`, `emunpu_device.c:282`):

```c
uint32_t chunk = emunpu_ceil_div(rows, cores);   // rows per core
for (each core ci) {
    start = ci * chunk;
    in.M        = <rows in this chunk>;
    in.a_offset = a_off + start*width*as;   // this core's input rows
    in.c_offset = c_off + start*cs;         // this core's output slots (disjoint)
    enqueue(cores[ci], in);
}
```

Because rows are independent, there is **no cross-core dependency**:

* each core reduces *only its own rows*, fully, start to finish;
* each core writes a *disjoint* slice of the output (`c_offset` shifted by `start`);
* there are **no partial sums to merge** — nothing to reconcile between cores.

The part that *is* sequentially dependent — the `acc += v` / `max` / `min` across a
row's `width` elements (`emunpu_core.c:400-417`) — happens **entirely within one
core**, in its own registers. That dependent inner loop is never split, so it never
needs a shared accumulator. This is the same strategy used everywhere in the
simulator: parallelize the embarrassingly-parallel axis (rows, output tiles, element
ranges) and keep the dependent axis local to a single core. The design **sidesteps**
the cross-core dependency rather than implementing a combine step.

### Two caveats

1. **A single huge reduction does not parallelize.** Reducing one row of large width
   (`rows == 1`) puts all the work on `core0`, reduced sequentially — there is **no
   tree / partial-sum reduction across cores**, precisely because the model relies on
   row-independence.
2. **`REDUCE` — like `SOFTMAX`, `NORM`, `CONV2D` and the rest of the ML op set —
   is a device-native API op, not on the IREE HAL path.** The coarse
   `emunpu-cmd-fb` ISA the compiler emits only has `matmul`/`add`/`relu`/`copy`, so
   in a *compiled* model a reduction (e.g. the sum/max inside a softmax) is
   **routed to the host CPU** in the heterogeneous flow ([`README.md`](README.md) §7)
   even though the hardware can do it. These ops live in the micro-ISA and are
   exercised by the device API, `ops_test`, and the CNN/transformer demos; see
   [`ISA.md` Appendix C](ISA.md#appendix-c--non-normative-proposed-v5-opcodes) for
   what exposing them to a compiler would take.
