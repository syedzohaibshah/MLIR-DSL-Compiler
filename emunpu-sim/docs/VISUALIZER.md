# Emulated NPU (`emunpu`) live visualizer

This document describes how the **multi-core NPU visualizer** works — the live
terminal dashboard that animates the 10 compute cores while a workload runs on
the `emunpu` device.

It is a companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (how the machine works)
and the multi-core architecture notes in
[`MULTICORE.md`](MULTICORE.md). The visualizer is
*observability only*: it never changes what the device computes — it just taps a
stream of activity events the engine emits and renders them.

---

## 1. What it is and why

Each matmul is split into 16×16 blocks and fanned (plus the DMA and elementwise
work) across `IREE_EMUNPU_NUM_CORES` (= 10) cores. For compiled programs that
16×16 tiling is decided by the compiler and the multi-core engine replays it; the
device-native API tiles on-device — either way the visualizer shows the same
runtime per-core fan-out. Text traces (`IREE_EMUNPU_TRACE=1`) prove *that* a
workload ran on the NPU, but they scroll past too fast to see *how the work is
distributed* across cores.

The visualizer answers that question live: which cores are busy, how the MMA /
DMA / vector load is balanced, how full global memory is, and how each GEMM fans
out into tiles — all updating as inference runs, in a **separate terminal** so it
doesn't interleave with the program's own output.

```
 EMUNPU — Multi-core NPU activity
 cores=10  active=10  util= 96%  instrs=18981  gemms=9  barriers=0
 ──────────────────────────────────────────────────────────────────────────────
 Global memory  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  64.5 KB / 4.0 MB
 Current GEMM   64x128x256 transpose_b  -> 32 tiles / 10 cores
 ──────────────────────────────────────────────────────────────────────────────
 core0  │ DMA.g2l     │ ████████████████████ 100% │ mma 696  dma 1446 vec 15  sfu 4    ●
 core1  │ DMA.g2l     │ ████████████████████ 100% │ mma 696  dma 1446 vec 6   sfu 4    ●
 core2  │ DMA.g2l     │ ███████████████████▏  86% │ mma 648  dma 1347 vec 6   sfu 4    ●
 ...
 core9  │ ACT.gelu    │ ███████████████████▏  86% │ mma 576  dma 1197 vec 6   sfu 4    ●
```

The per-core **bar is live utilization** — each core's share of the *current
operation's* work, not a lifetime total — so a GEMM tiled across all cores lights
every participating core at once (`util=` is the fleet average, `active=` the
count above 5%). A single-core run, or an op with less work than there are cores,
leaves the other bars empty. See §4 for how that's reconstructed from a serial
event stream.

---

## 2. Design: a one-way event stream

The visualizer is split into a **producer** (C, inside the engine) and a
**consumer** (a standalone Python script). They are decoupled by a plain text
file:

```
        engine (producer)                         viewer (consumer)
   runtime/.../emunpu_viz.{h,c}                       emunpu_demos/emunpu_viz.py
            │  appends one line per event                  ▲  tails the file,
            ▼                                               │  parses lines,
     ┌───────────────────────────┐   read (tail -f style)  │  renders a frame
     │  $IREE_EMUNPU_VIZ file     │ ───────────────────────►│  at a fixed FPS,
     │  (e.g. /tmp/emunpu.events) │                         │  op-by-op
     └───────────────────────────┘
```

Why a file rather than a socket/FIFO:

* **No blocking / no ordering hazard.** The producer only ever appends and
  flushes; the consumer only reads. Neither blocks the other, and it doesn't
  matter which starts first.
* **Works over SSH and across processes.** The model can run under
  `iree-run-module`, a PyTorch script, or the standalone test — anything that
  links the engine — and the viewer is a separate process you can put on another
  pane/tmux window/SSH session.
* **Replayable.** The file is a full record of the run; `--replay` re-renders it
  later.

Emission is gated by the environment variable: if `IREE_EMUNPU_VIZ` is unset the
producer does nothing (one cached `getenv` check), so ordinary runs pay no cost.

---

## 3. Quick start (two terminals)

```bash
# Terminal 1 — start the viewer first (it creates/owns the event file):
python emunpu_demos/emunpu_viz.py /tmp/emunpu.events

# Terminal 2 — run any emunpu workload pointing IREE_EMUNPU_VIZ at the same file:
IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_demos/emunpu_pytorch_multicore.py --iters 20
# ...or the CLI:
IREE_EMUNPU_VIZ=/tmp/emunpu.events \
  ../iree-build/tools/iree-run-module --device=emunpu --module=mm.vmfb \
  --function=matmul --input="96x96xf32=1.0" --input="96x96xf32=1.0"
```

`emunpu_demos/emunpu_pytorch_multicore.py` defaults `IREE_EMUNPU_VIZ` to `/tmp/emunpu.events`
if you don't set it, so for that demo you only have to point the viewer there.

> Use a model with layers **larger than 16×16** to light up all the cores. Tiny
> matmuls (≤ 16×16) are a single tile and only `core0` runs — that's correct, not
> a bug. `emunpu_demos/emunpu_pytorch_multicore.py` is sized for the multi-core view; the
> original `emunpu_demos/emunpu_pytorch.py` is intentionally tiny (single-tile).

### One core vs. many — the utilization contrast

The number of online cores is set by `IREE_EMUNPU_NUM_CORES` (1..10), read by the
runtime when it creates the device. The compiler always bakes a fixed 10-core
round-robin tile schedule; the runtime *folds* those core ids onto however many
cores are online (`emunpu_run_gemm_schedule`), so the **same compiled binary**
runs on any core count. Two demos make the contrast obvious side-by-side against
the same workload:

```bash
# Many cores (default 10) — every large GEMM fans across all cores at once:
IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_demos/emunpu_pytorch_multicore.py --iters 20
#   ...or an explicit count:  --cores 4

# One core — identical workload, but every tile lands on core0 (core0 pinned
# near 100%, the rest idle):
IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_demos/emunpu_singlecore.py --iters 20
```

Watch both through the same viewer and the *only* difference is how utilization
spreads across the core rows.

The viewer paces itself with `--rate` (operations revealed per second, default 6)
and `--fps` (frame rate, default 30) so a fast run animates op-by-op with the
bars easing smoothly, rather than the whole run flashing past in one frame.

---

## 4. The event schema

Events are line-oriented, `TAG key=value ...`, one per line. The producer flushes
after every line so the viewer sees them promptly.

| Tag | Emitted when | Fields | Meaning |
|-----|--------------|--------|---------|
| `RUN` | device created | `cores`, `cap` | start of a run; viewer resets its state |
| `GEMM` | a matmul is scheduled | `m n k t tiles cores` | GEMM dims, transpose-B flag `t`, tile count, #cores |
| `INSTR` | a core executes an instruction | `core kind dt m n k f` [`sub`] | per-core op: `kind` ∈ DMA.g2l/DMA.l2g/MMA16/MATVEC/EW/REDUCE/FMA/CONVERT/ACT/SOFTMAX/NORM/TRANSPOSE/IM2COL/POOL2D/GATHER. `sub` is present only for ops with a sub-operation and names it (`sub=gelu`, `sub=add`, `sub=layernorm`, `sub=max`, …) |
| `MEM` | after a command runs, and once at process exit | `used`, `cap` | global memory bytes in use / fixed capacity; the exit event reports the freed arena so the viewer's bar drains |
| `BARRIER` | device barrier | `streams` | a cross-stream/-core barrier was hit |

Example fragment (a 40×48×24 GEMM, first output tile on `core0`, K=24 → two
K-tiles 16+8):

```
RUN cores=10 cap=268435456
GEMM m=40 n=48 k=24 t=0 tiles=9 cores=10
INSTR core=0 kind=DMA.g2l dt=fp32 m=16 n=16 k=0 f=0
INSTR core=0 kind=DMA.g2l dt=fp32 m=16 n=16 k=0 f=0
INSTR core=0 kind=MMA16   dt=fp32 m=16 n=16 k=16 f=0
INSTR core=0 kind=DMA.g2l dt=fp32 m=16 n=8  k=0 f=0
INSTR core=0 kind=DMA.g2l dt=fp32 m=8  n=16 k=0 f=0
INSTR core=0 kind=MMA16   dt=fp32 m=16 n=16 k=8  f=2     # f bit 2 = accumulate
INSTR core=0 kind=DMA.l2g dt=fp32 m=16 n=16 k=0 f=0
INSTR core=1 kind=DMA.g2l ...
MEM used=8448 cap=268435456
```

An ML-op fragment (a layernorm's normalize + gamma multiply, then a GELU), where
the `sub` field names the sub-operation the `flags` value selects:

```
INSTR core=0 kind=NORM    dt=fp32 m=3  n=32 k=0 f=0 sub=layernorm
INSTR core=0 kind=EW      dt=fp32 m=96 n=0  k=0 f=2 sub=mul
INSTR core=0 kind=ACT     dt=fp32 m=154 n=0 k=0 f=6 sub=gelu
INSTR core=0 kind=POOL2D  dt=fp32 m=6  n=0  k=0 f=0 sub=max
INSTR core=0 kind=SOFTMAX dt=fp32 m=3  n=24 k=0 f=0
```

---

## 5. Producer side (`runtime/src/iree/hal/drivers/emunpu/emunpu_viz.{h,c}`)

A tiny, IREE-independent module (stdio only). Public API:

| Function | Called from |
|----------|-------------|
| `emunpu_viz_enabled()` | cached `getenv("IREE_EMUNPU_VIZ")` check |
| `emunpu_viz_run_begin(cores, cap)` | `emunpu_device_create` (emits `RUN`) |
| `emunpu_viz_gemm(m,n,k,t,tiles,cores)` | a GEMM dispatch in `emunpu_device.c` — the on-device scheduler (`emunpu_schedule_gemm`) or the compiler-schedule replay (`emunpu_matmul_scheduled`); both emit `GEMM` |
| `emunpu_viz_instr(core_id, instr)` | `emunpu_core_execute` — every instruction (emits `INSTR`) |
| `emunpu_viz_mem(used, cap)` | after each command runs, and from an at-exit drain in `emunpu_hw.c` (emits `MEM`) |
| `emunpu_viz_barrier(streams)` | `emunpu_device_barrier` (emits `BARRIER`) |

Key behaviors:

* **Lazy open, truncate-once.** The file is opened `"w"` (truncated) the first
  time an event is emitted in a process, then appended to. So re-running a
  workload starts the file fresh; the viewer detects the truncation and resets
  (§6).
* **Line-buffered + per-line flush** so the tailing viewer never waits on a
  stale buffer.
* **No-op when disabled.** Every emit function early-returns if
  `IREE_EMUNPU_VIZ` is unset — zero overhead on normal runs.

Because emission lives in `emunpu_core_execute` (per instruction) and the device
layer (run/gemm/mem/barrier), it captures *all* device activity regardless of
how the work arrived — the HAL matmul/elementwise paths, or direct device-API
calls.

---

## 6. Consumer side (`emunpu_demos/emunpu_viz.py`)

A single dependency-free Python script (only ANSI escapes — no curses, no
`rich`). Ingestion and rendering are **decoupled**: events are read and metered
into "waves," while the frame is redrawn on a fixed clock.

**Tail loop.** Polls the file size; reads only the bytes appended since the last
position, buffers partial lines, parses complete lines into a `pending` queue. If
the file *shrinks* (a new run truncated it) it rewinds to the start and resets.

**Reconstructing concurrency (the key idea).** The engine is single-threaded, so
it emits *all* of core0's instructions, then all of core1's, and so on — the
stream is serialized even though the hardware model treats a GEMM's tiles as
running concurrently across the cores. Rendering one row per event would light
one core at a time and look sequential. Instead the viewer groups events into
**waves**: one coarse op (a GEMM, or an elementwise add/relu/copy) is a wave,
delimited by the `MEM` event the engine emits at the end of every op. When a wave
finalizes, each core's utilization target is set to *its instruction count in the
wave ÷ the busiest core's count* — so a balanced GEMM gives every participating
core ≈100% at the same time, an op with fewer work units than cores lights only
the first few, and a 1-core run gives `core0` alone.

**Model.** Per core: dominant op class of the current wave, the last op mnemonic
seen (with its sub-operation), cumulative `mma`/`dma`/`vec`/`sfu` counts, a `util`
target and an eased `disp_util` actually drawn. Plus the current GEMM, global
memory, and totals. `RUN` resets; `GEMM` opens a wave; `INSTR` accumulates into
it; `MEM` finalizes it.

**Renderer.** At a fixed FPS it eases each `disp_util` toward its target (fast
up, slower down), then prints a frame: header (`cores`, `active`, fleet-average
`util`, totals), global-memory bar, current GEMM, then one row per core — the
current op, a smooth fractional-block utilization bar with a percentage,
cumulative mma/dma/vec/sfu counts, and a `●` busy pulse. When the stream goes
quiet the bars ease back to idle.

Options:

| Flag | Effect |
|------|--------|
| `path` (positional) | event file to tail (default `/tmp/emunpu.events`) |
| `--fps N` | render frames per second (default `30`) |
| `--rate N` | operations (waves) revealed per second (default `6`; `0` = as fast as read) |
| `--replay` | replay an existing file once from the start, then exit (for a captured run) |

A fast workload emits thousands of events almost instantly; `--rate` meters how
many *operations* are revealed per second so the run animates op-by-op instead of
snapping to the end in a single frame. Lower it for slow-motion, raise it (or set
`0`) to fast-forward.

---

## 7. Reading the dashboard — what each figure shows

The display is four blocks, top to bottom. Here is every figure and exactly what
it means.

```
 EMUNPU — Multi-core NPU activity                                   <- title
 cores=10  active=10  util= 96%  instrs=18981  gemms=9  barriers=0   <- header
 ────────────────────────────────────────────────────────────────
 Global memory  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  64.5 KB / 4.0 MB    <- memory bar
 Current GEMM   64x128x256 transpose_b  -> 32 tiles / 10 cores       <- current GEMM
 ────────────────────────────────────────────────────────────────
 core0  │ DMA.g2l     │ ████████████████████ 100% │ mma 696 dma 1446 vec 15 sfu 4 ●  <- per-core row
 ...
```

* **Title** — `EMUNPU — Multi-core NPU activity`. Static banner.

* **Header line** — one-glance fleet summary:
  * `cores` — how many cores are **online** (set by `IREE_EMUNPU_NUM_CORES`,
    1..10). This is the number of core rows drawn below.
  * `active` — cores above 5% utilization **right now** (this frame).
  * `util` — the **fleet-average** utilization (mean of all core bars) as a
    percent; ~100% means the current op is spread evenly across every core.
  * `instrs` — cumulative count of every core instruction executed so far
    (DMA + MMA + vector), across the whole run.
  * `gemms` — cumulative number of matmuls (GEMM headers) seen.
  * `barriers` — cumulative cross-core/stream barriers hit.

* **Global memory bar** — occupancy of the NPU's **fixed 4 MB global memory**
  (`used / 4.0 MB`), the single arena all tensors share. The bar and its
  denominator are a real hardware size (`IREE_EMUNPU_GLOBAL_MEM_BYTES`), *not*
  sized to the workload, so a small model reads a small fraction (e.g.
  `64.5 KB / 4.0 MB`) and a large one fills more of it. `used` is the live
  working set of the op currently executing (its input + output tensors); it is
  sampled while the op runs, so the bar shows each op's **peak** footprint.
  Tensors are freed back to the coalescing free-list after every op, and when the
  **whole workload ends** the bar **drains to (near) empty** — an at-exit event
  reports the final freed occupancy. (If an op's working set ever exceeds 4 MB
  the arena grows just for it and the denominator rises to match.)

* **Current GEMM** — the most recent matmul: `MxNxK`, a `transpose_b` tag when
  the RHS is transposed (`nn.Linear`), and how many 16×16 output tiles it fanned
  into across how many cores. `(none yet)` before the first matmul.

* **Per-core row** — one per online core:
  * `core<i>` — the core id (0-based).
  * **current op** — the instruction the core last executed in this wave, with
    its sub-operation (`MMA16`, `DMA.g2l`, `EW.add`, `ACT.gelu`, `POOL2D.max`,
    `NORM.layernor…`), or `idle`. Its colour is the op *class*: `MMA`
    (tensor-core), `DMA` (global↔local transfer), `VEC` (elementwise / reduction
    / data marshalling), `SFU` (activation, softmax, normalization). GEMM tiling
    is DMA-heavy, so a matmul wave usually reads `DMA` — honest: each 16×16 tile
    is 2 DMA-in + 1 MMA + a DMA-out.
  * **utilization bar + %** — this core's share of the *current op's* work
    relative to the busiest core (live, not lifetime). Even bars near 100% = a
    GEMM fanned evenly across cores; a single filled bar = a single-tile op, an
    op with only one work unit, or a 1-core run. Colour goes cool→hot with load.
  * `mma` / `dma` / `vec` / `sfu` — **cumulative** tensor-core, DMA, vector
    (elementwise / reduction / transpose / im2col / pooling / gather), and
    special-function (activation / softmax / normalization) instruction counts on
    that core over the whole run.
  * `●` — busy pulse, shown when the bar is above ~35% (colour tracks
    utilization). Watch it ripple across cores as a GEMM fans out.

* **Legend line** — a one-line reminder at the bottom (`bar = live utilization …`).

What "good" looks like: for a large GEMM all online bars fill roughly equally
(the round-robin split leaves at most a one-tile gap, so the ~86% vs 100% split
above); `mma`/`dma` track the matmul tiles, `vec` tracks the bias/elementwise
work, and `sfu` tracks activations, softmax, and normalization. Broadcast ops
(bias add, layernorm's gamma/beta) split on period boundaries, so they fan across
the cores too rather than pinning to `core0`.

The `cnn_demo` and `transformer_demo` workloads are the most interesting things to
watch: the CNN alternates `IM2COL` → `MMA16` waves with `POOL2D`, and the
transformer block cycles `NORM` → `MMA16` → `SOFTMAX` → `ACT.gelu`, so all four
op classes light up in turn.

---

## 8. Worked example: single-core vs. multi-core

The two demo scripts are **standalone** and the core usage is decided by their
**model dimensions** — that is the whole point of the pair:

```bash
# Single-core: small Linear stack, every dim <=16 => 1 tile/layer => only core0.
IREE_EMUNPU_VIZ=/tmp/e1 python emunpu_demos/emunpu_singlecore.py --iters 20
python emunpu_demos/emunpu_viz.py /tmp/e1 --replay        # only core0 fills

# Multi-core: large MLP (batch 64, hidden 256) => dozens of tiles => all cores.
IREE_EMUNPU_VIZ=/tmp/e2 python emunpu_demos/emunpu_pytorch_multicore.py --iters 20
python emunpu_demos/emunpu_viz.py /tmp/e2 --replay        # all 10 cores ~equal
```

Why the dimensions decide it: a `Linear(K, N)` on a batch of `M` rows is an
`M x N x K` matmul, tiled into `ceil(M/16) * ceil(N/16)` output tiles that the
scheduler round-robins across the cores. Keep every dimension `<= 16` (the
single-core script's defaults) and each layer is a single tile that lands on
`core0`; grow them past 16 (the multi-core script's defaults) and each matmul
fans into many tiles across all the cores. Both scripts print the tile math up
front, e.g. `layer0: GEMM 64x256x128 -> 64 output tiles x 8 K-tiles`, so the
dashboard's utilization spread matches the predicted fan-out.

> The single-core model is Linear-only (no ReLU): elementwise and activation ops
> *always* range-split across every core regardless of matmul tiling, so a ReLU
> would light up all the cores. With every dimension ≤ 16 the matmul is a single
> tile on `core0` and a broadcast bias is a single period-aligned chunk, so the
> Linear-only stack is genuinely single-core.
>
> Independently of dimensions, `IREE_EMUNPU_NUM_CORES` (1..10) caps how many
> cores are *online* — the multi-core script's optional `--cores N` sets it, and
> the compiler's 10-core tile schedule is folded onto that many cores at replay.

---

## 9. Troubleshooting

* **Only `core0` is busy** — either your matmuls are ≤ 16×16 (single tile; use
  bigger layers / batch, e.g. `emunpu_demos/emunpu_pytorch_multicore.py`), or the
  device only brought up one core (`IREE_EMUNPU_NUM_CORES=1`, as
  `emunpu_singlecore.py` does). The header's `cores=` shows how many are online.
* **Nothing appears** — confirm `IREE_EMUNPU_VIZ` is set *in the workload's*
  environment (terminal 2) and points at the same path the viewer is tailing.
* **The view resets mid-run** — expected only if a single op's working set
  exceeds the fixed 4 MB global memory, forcing the device to grow (it recreates
  the device, which re-emits `RUN`). Ordinary workloads create the device once at
  the fixed 4 MB and never regrow.
* **The memory bar stays full after the run finishes** — it drains on process
  exit (the engine emits a final `MEM` reflecting the freed arena). If you killed
  the workload with SIGKILL, the at-exit handler never ran, so the last in-run
  peak is all the viewer ever saw — rerun and let it exit cleanly.
* **Garbled colors** — the terminal must support ANSI escapes; pipe through a
  real TTY (not a pager).

---

## 10. Extending

To surface a new kind of activity:

1. Add an emit function + event tag in `emunpu_viz.{h,c}` (keep it append-only
   and flush per line).
2. Call it from the relevant engine site (a core kernel, a scheduler decision, a
   device API entry point).
3. Handle the new tag in `emunpu_demos/emunpu_viz.py`'s `Model.apply` and show it in
   `render`.

The schema is deliberately loose (`TAG key=value`), so adding fields/tags never
breaks an older viewer — unknown tags are simply ignored.
