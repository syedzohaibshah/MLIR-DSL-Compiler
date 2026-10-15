#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Live terminal visualizer for the emulated multi-core NPU.

Renders, in real time, the activity of the compute cores while a workload runs
on the `emunpu` device: which core is executing which class of instruction, the
*live* per-core utilization, the op mix (MMA / DMA / vector), global-memory
occupancy, and GEMM tiling progress.

It works by tailing the structured event stream the engine writes when the
environment variable IREE_EMUNPU_VIZ names a file (see emunpu_viz.c). No
third-party dependencies -- just ANSI escapes -- so it runs over SSH.

How the display stays honest about parallelism
----------------------------------------------
The engine is single-threaded, so it emits every instruction of core0, then
every instruction of core1, and so on -- the events are serialized even though
the hardware model treats the cores as running *concurrently* within one op.
Animating one row per event would light up a single core at a time and look
sequential even for a fully-parallel GEMM.

Instead the visualizer groups events into "waves": one coarse operation (a
GEMM, or an elementwise add/relu/copy) is a wave, delimited by the MEM event the
engine emits at the end of every op. All cores that did work in a wave are shown
busy *together*, with each core's utilization set to its share of that wave's
work relative to the busiest core. So:

  * a GEMM tiled across all cores  -> every participating core bar near 100%
  * a single-tile / broadcast op   -> only core0 lights up
  * a 1-core run                   -> core0 at 100%, the rest idle

Rendering is decoupled from ingestion: waves are admitted at a metered rate and
the frame is redrawn at a fixed FPS with the bars easing smoothly toward their
targets, so bursty runs animate op-by-op instead of snapping.

Usage (two terminals):

  # Terminal 1 -- start the visualizer (it creates/owns the event file):
  python emunpu_viz.py /tmp/emunpu.events

  # Terminal 2 -- run any emunpu workload, pointing IREE_EMUNPU_VIZ at it:
  IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_pytorch_multicore.py
  IREE_EMUNPU_VIZ=/tmp/emunpu.events python emunpu_singlecore.py

Options:
  path            event file to tail (default /tmp/emunpu.events)
  --fps N         render frames per second (default 30)
  --rate N        operations (waves) revealed per second, so a fast run animates
                  op-by-op instead of snapping (default 6; 0 = as fast as read)
  --replay        replay an existing file from the start once, then exit
"""

import argparse
import collections
import os
import sys
import time

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
HIDE_CUR = "\x1b[?25l"
SHOW_CUR = "\x1b[?25h"
CLEAR_HOME = "\x1b[2J\x1b[H"

# Utilization gradient (cool idle -> hot saturated).
HEAT_COLORS = ["\x1b[90m", "\x1b[34m", "\x1b[36m", "\x1b[32m", "\x1b[33m",
               "\x1b[91m"]
# Per-op-class accent colors used for the current-op label.
OP_COLORS = {"MMA": "\x1b[95m", "DMA": "\x1b[36m", "VEC": "\x1b[33m",
             "SFU": "\x1b[92m", "OTHER": DIM}
# Fractional block glyphs for smooth (sub-cell) bars.
BLOCKS = " ▏▎▍▌▋▊▉█"


def heat_color(frac):
    frac = max(0.0, min(1.0, frac))
    idx = min(len(HEAT_COLORS) - 1, int(frac * (len(HEAT_COLORS) - 1) + 0.001))
    return HEAT_COLORS[idx]


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def bar(frac, width):
    """A smooth bar using fractional block glyphs for the trailing cell."""
    frac = max(0.0, min(1.0, frac))
    total = frac * width
    full = int(total)
    out = "█" * full
    if full < width:
        out += BLOCKS[int(round((total - full) * 8))]
        out += " " * (width - full - 1)
    return out[:width]


# Vector-engine ops (elementwise / data marshalling) and special-function-unit
# ops (transcendental activations, softmax, normalization) are counted apart so
# an activation-heavy stage is visibly different from a pure elementwise one.
VEC_OPS = ("EW", "REDUCE", "FMA", "CONVERT", "MATVEC", "TRANSPOSE", "IM2COL",
           "POOL2D", "GATHER")
SFU_OPS = ("ACT", "SOFTMAX", "NORM")


def op_class(kind):
    """Buckets an engine op name into MMA / DMA / VEC / SFU / OTHER."""
    if kind.startswith("MMA"):
        return "MMA"
    if kind.startswith("DMA"):
        return "DMA"
    if kind in VEC_OPS:
        return "VEC"
    if kind in SFU_OPS:
        return "SFU"
    return "OTHER"


class CoreState:
    __slots__ = ("cur_class", "cur_op", "mma", "dma", "vec", "sfu", "other",
                 "total", "util", "disp_util")

    def __init__(self):
        self.cur_class = None   # dominant op class in the current wave
        self.cur_op = None      # last op mnemonic (+ sub-op) seen this wave
        self.mma = self.dma = self.vec = self.sfu = self.other = self.total = 0
        self.util = 0.0         # target utilization for the current wave [0,1]
        self.disp_util = 0.0    # eased value actually drawn (for smoothness)


class Model:
    def __init__(self, num_cores=10):
        self.reset(num_cores, 0)

    def reset(self, num_cores, cap):
        self.num_cores = max(1, num_cores)
        self.cap = cap
        self.cores = [CoreState() for _ in range(self.num_cores)]
        self.mem_used = 0
        self.gemm = None          # (m, n, k, t, tiles, cores)
        self.gemm_count = 0
        self.barrier_count = 0
        self.total_instr = 0
        self.last_core = -1
        # Accumulator for the wave (single coarse op) currently being decoded.
        self._wave_count = collections.Counter()          # core -> instrs
        self._wave_ops = collections.defaultdict(collections.Counter)  # core->{cls}

    def _finalize_wave(self):
        """Turns the accumulated wave into per-core utilization targets: the
        busiest core in the wave reads 100%, the rest scale by their share, and
        cores that did nothing this wave fall to 0. This is what makes every
        core of a parallel GEMM light up together despite the serial stream."""
        if not self._wave_count:
            return
        peak = max(self._wave_count.values())
        for i, cs in enumerate(self.cores):
            n = self._wave_count.get(i, 0)
            cs.util = (n / peak) if peak else 0.0
            ops = self._wave_ops.get(i)
            cs.cur_class = ops.most_common(1)[0][0] if ops else cs.cur_class
        self._wave_count = collections.Counter()
        self._wave_ops = collections.defaultdict(collections.Counter)

    def apply(self, tag, kv):
        if tag == "RUN":
            self.reset(int(kv.get("cores", 10)), int(kv.get("cap", 0)))
        elif tag == "GEMM":
            # A GEMM header opens a new op; flush anything still pending first.
            self._finalize_wave()
            self.gemm = (int(kv["m"]), int(kv["n"]), int(kv["k"]),
                         int(kv.get("t", 0)), int(kv.get("tiles", 0)),
                         int(kv.get("cores", self.num_cores)))
            self.gemm_count += 1
        elif tag == "INSTR":
            c = int(kv["core"])
            if 0 <= c < self.num_cores:
                cs = self.cores[c]
                kind = kv.get("kind", "OP")
                cls = op_class(kind)
                sub = kv.get("sub")
                cs.cur_op = f"{kind}.{sub}" if sub else kind
                if cls == "MMA":
                    cs.mma += 1
                elif cls == "DMA":
                    cs.dma += 1
                elif cls == "VEC":
                    cs.vec += 1
                elif cls == "SFU":
                    cs.sfu += 1
                else:
                    cs.other += 1
                cs.total += 1
                self.last_core = c
                self.total_instr += 1
                self._wave_count[c] += 1
                self._wave_ops[c][cls] += 1
        elif tag == "MEM":
            # End of a coarse op -> commit the wave's utilization.
            self.mem_used = int(kv.get("used", 0))
            self.cap = int(kv.get("cap", self.cap))
            self._finalize_wave()
        elif tag == "BARRIER":
            self.barrier_count += 1

    def decay_idle(self, factor=0.6):
        """Fades utilization toward 0 when the workload has gone quiet."""
        for cs in self.cores:
            cs.util *= factor

    def ease(self, rise=0.5, fall=0.18):
        """Eases the drawn bar toward the target (fast up, slower down)."""
        for cs in self.cores:
            a = rise if cs.util >= cs.disp_util else fall
            cs.disp_util += (cs.util - cs.disp_util) * a
            if cs.disp_util < 0.005:
                cs.disp_util = 0.0

    def active_now(self):
        return sum(1 for cs in self.cores if cs.disp_util > 0.05)

    def aggregate_util(self):
        if not self.cores:
            return 0.0
        return sum(cs.disp_util for cs in self.cores) / len(self.cores)


def render(model, width=88):
    out = [CLEAR_HOME]
    out.append(f" {BOLD}EMUNPU — Multi-core NPU activity{RESET}")
    out.append(f" {DIM}cores={model.num_cores}  active={model.active_now()}  "
               f"util={model.aggregate_util() * 100:4.0f}%  "
               f"instrs={model.total_instr}  gemms={model.gemm_count}  "
               f"barriers={model.barrier_count}{RESET}")
    out.append(" " + "─" * width)

    # Global memory bar.
    frac = (model.mem_used / model.cap) if model.cap else 0.0
    out.append(f" {BOLD}Global memory{RESET}  [{bar(frac, 30)}]  "
               f"{human_bytes(model.mem_used)} / {human_bytes(model.cap)}")

    # Current GEMM.
    if model.gemm:
        m, n, k, t, tiles, cores = model.gemm
        tb = " transpose_b" if t else ""
        out.append(f" {BOLD}Current GEMM{RESET}   {m}x{n}x{k}{tb}  "
                   f"-> {tiles} tiles / {cores} cores")
    else:
        out.append(f" {BOLD}Current GEMM{RESET}   {DIM}(none yet){RESET}")
    out.append(" " + "─" * width)

    # Per-core rows. The bar is *live* utilization (share of the current op's
    # work), not a lifetime total, so a balanced GEMM lights every participating
    # core simultaneously and idle cores read empty.
    for i, cs in enumerate(model.cores):
        col = heat_color(cs.disp_util)
        load = bar(cs.disp_util, 20)
        pct = f"{cs.disp_util * 100:3.0f}%"
        if cs.disp_util > 0.05 and cs.cur_class:
            kcol = OP_COLORS.get(cs.cur_class, DIM)
            label = f"{kcol}{(cs.cur_op or cs.cur_class)[:11]:<11}{RESET}"
        else:
            label = f"{DIM}{'idle':<11}{RESET}"
        pulse = f"{col}●{RESET}" if cs.disp_util > 0.35 else " "
        out.append(
            f" {col}core{i:<2}{RESET} │ {label} │ "
            f"{col}{load}{RESET} {pct} │ "
            f"mma {cs.mma:<5} dma {cs.dma:<5} vec {cs.vec:<4} "
            f"sfu {cs.sfu:<4} {pulse}")

    out.append(" " + "─" * width)
    out.append(f" {DIM}legend: bar = live utilization (share of current op), "
               f"color/● = busy. Ctrl-C to quit.{RESET}")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def parse_line(line):
    parts = line.split()
    if not parts:
        return None, {}
    tag = parts[0]
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            key, _, val = p.partition("=")
            kv[key] = val
    return tag, kv


def main():
    ap = argparse.ArgumentParser(description="Live emunpu multi-core visualizer")
    ap.add_argument("path", nargs="?", default="/tmp/emunpu.events",
                    help="event file written by the engine (IREE_EMUNPU_VIZ)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="render frames per second (default 30)")
    ap.add_argument("--rate", type=float, default=6.0,
                    help="operations (waves) revealed per second so a fast run "
                         "animates op-by-op (default 6; 0 = as fast as read)")
    ap.add_argument("--replay", action="store_true",
                    help="replay an existing file once from the start, then exit")
    args = ap.parse_args()

    frame_dt = 1.0 / max(1.0, args.fps)

    model = Model()
    # Create the file fresh so a producer's truncate-on-open and our tail agree.
    if not args.replay:
        try:
            open(args.path, "w").close()
        except OSError:
            pass

    sys.stdout.write(HIDE_CUR)
    pos = 0
    buf = ""
    pending = collections.deque()   # parsed (tag, kv) events awaiting metering
    wave_budget = 0.0               # fractional waves we're allowed to reveal
    idle_frames = 0
    try:
        while True:
            # --- ingest: pull any newly-appended bytes into the pending queue.
            try:
                size = os.path.getsize(args.path)
            except OSError:
                size = 0
            if size < pos:  # producer truncated -> new run
                pos = 0
                buf = ""
                pending.clear()
                model.reset(model.num_cores, model.cap)
            if size > pos:
                with open(args.path, "r") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                buf += chunk
                while "\n" in buf:
                    line, _, buf = buf.partition("\n")
                    tag, kv = parse_line(line)
                    if tag:
                        pending.append((tag, kv))

            # --- meter: reveal at most `wave_budget` whole operations (each
            # ended by a MEM event) this frame; carry the remainder over.
            if args.rate > 0:
                wave_budget += args.rate * frame_dt
            else:
                wave_budget = float(len(pending))  # unmetered: drain all
            waves_done = 0
            consumed = False
            while pending and waves_done < int(wave_budget) + (args.rate <= 0):
                tag, kv = pending.popleft()
                model.apply(tag, kv)
                consumed = True
                if tag == "MEM":
                    waves_done += 1
            wave_budget = max(0.0, wave_budget - waves_done)

            # --- animate + render at a steady cadence.
            if consumed:
                idle_frames = 0
            else:
                idle_frames += 1
                if idle_frames > 2:      # workload quiet -> ease bars back down
                    model.decay_idle()
            model.ease()
            render(model)

            if args.replay and not pending and idle_frames > int(args.fps):
                break
            time.sleep(frame_dt)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW_CUR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
