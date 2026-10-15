# Integrating the emulated NPU into a compiler stack

This folder shows how the portable simulator (`../include` + `../src`) was wired
behind a real compiler and runtime (IREE). Use it as a worked reference when
attaching your own backend. The files here **require IREE to compile** and are
not built by this project's Makefile/CMake — they are here to read, not to link.

## The three layers

```
   your compiler backend                       your runtime / HAL driver
   ─────────────────────                       ─────────────────────────
   recognizes matmul/add/relu/copy             loads the blob, decodes commands,
   tiles matmul into 16×16 blocks   ── blob ─▶ stages operands into global memory,
   emits a command + tile-schedule blob        replays the schedule on the device
                    │                                          │
                    └────────── shared contract ───────────────┘
                          reference/emunpu_isa.h  (the "ISA")
                                       │
                                       ▼
                          the hardware model you drive
                        ../include/emunpu/device.h
```

1. **Hardware model** — `../include` + `../src`, portable C. Your runtime drives
   it like a CUDA/HIP driver: `emunpu_device_create`, `alloc`, `upload`,
   `emunpu_matmul_scheduled` / `emunpu_gemm` / `emunpu_vector_op`, `download`.

2. **The ISA / blob contract** — [`emunpu_isa.h`](emunpu_isa.h), formally
   specified in [`../docs/ISA.md`](../docs/ISA.md). The single source of truth
   for the binary your code generator emits and your runtime decodes. It is plain
   C POD structs (C/C++ compatible, dependency-free) so both halves `#include`
   the same file. A dispatch is a small array of coarse commands
   (`MATMUL`/`ADD`/`RELU`/`COPY`) referencing dispatch bindings by ordinal, plus
   — for matmul — a **compile-time tile schedule** (`emunpu_matmul_tile_t[]`):
   the 16×16 block grid and per-core assignment, decided by the compiler exactly
   like a GPU backend bakes in a workgroup grid. See the spec for the byte-level
   layout, encoding tables, and a worked fused-dispatch example.

3. **The runtime glue** — [`emunpu_hw.c`](emunpu_hw.c) /
   [`emunpu_loader.c`](emunpu_loader.c). `emunpu_hw.c` turns one decoded command
   into device-API calls against a shared lazy-singleton device (matmul → replay
   the tile schedule via `emunpu_matmul_scheduled`; elementwise → `emunpu_vector_op`).
   `emunpu_loader.c` plugs into IREE's `local-sync` executable-loader interface:
   it validates the blob header and dispatches commands. `emunpu_mem.fbs` /
   `emunpu_mem_fb.*` serialize the global-memory layout via FlatBuffers.

## What to build for your own stack

- **Compiler side:** recognize the coarse ops in your IR, decide the 16×16
  matmul tiling (mirror `buildMatmulTileSchedule` from the original target
  backend), and serialize an `emunpu-cmd-fb` blob per the layout documented at
  the top of `emunpu_isa.h`. Both halves must agree on `IREE_EMUNPU_MMA_DIM`
  and `IREE_EMUNPU_NUM_CORES` (see `../include/emunpu/arch.h`) — those are
  the physical device parameters you tile against.

- **Runtime side:** decode the blob (header → exports → commands → tiles →
  string table), stage each binding into global memory with
  `emunpu_device_alloc` + `emunpu_device_upload`, execute each command against
  the device, then `emunpu_device_download` the results. `emunpu_hw.c` is a
  ~complete template for the per-command execution.

## Linking the hardware model

The hardware model (`../include` + `../src`) builds as a **shared object** —
`libemunpu.so` (`.dylib` / `emunpu.dll`) — and also a static archive. Only the
public device/runtime API is exported: every function in the headers is tagged
`EMUNPU_API` (defined in [`../include/emunpu/export.h`](../include/emunpu/export.h))
and the library is compiled with hidden default visibility, so internal helpers
(all `static`) never reach the ABI. Verify the exported surface with
`nm -D --defined-only build/libemunpu.so.1 | grep ' T '` — every symbol is
`emunpu_*`.

Your runtime links it exactly as it would a CUDA/HIP driver library:

```bash
# after `cmake --install build --prefix <prefix>` (installs .so + headers +
# a find_package package under <prefix>/lib/cmake/emunpu):
cc your_runtime.c -I<prefix>/include -L<prefix>/lib -lemunpu -o your_runtime
```

From a CMake project, `find_package(emunpu)` then link the imported
`emunpu::emunpu` target — it carries the include dirs and the `.so`. When you
instead link the **static** archive, define `EMUNPU_STATIC` so `EMUNPU_API`
drops its import/export decoration (CMake's `EMUNPU_BUILD_SHARED=OFF` sets this
for you). Nothing else about the integration changes — the loader/HAL glue below
calls the same API whether it resolves against the shared or static build.

## Swapping the model

`emunpu_hw.c` is intentionally the only file that couples the ISA to the device
API, so you can replace the hardware model (cycle-accurate simulator,
out-of-process model, or real silicon) without touching the loader/HAL wiring —
keep the command-execution signature and re-target its body.

For the full design rationale (memory subsystem, scheduler, dtype handling,
multi-core fan-out) read [`../docs/MULTICORE.md`](../docs/MULTICORE.md) and
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). File paths in
those docs refer to the original IREE source tree.
