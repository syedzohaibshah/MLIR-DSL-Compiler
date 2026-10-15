# emunpu-sim — Emulated multi-core NPU simulator

A self-contained, software-modeled **multi-core NPU** you can build, test, run,
and visualize with nothing but a C compiler — no IREE, no ML framework, no
Python packages. It is extracted from a working IREE backend so you can drop the
*hardware model* into your own compiler/runtime stack and wire a code-generator
against it.

The device models a realistic coarse-grained accelerator:

- a centralized **global memory** subsystem (allocator + host↔device DMA),
- **10 compute cores** (`IREE_EMUNPU_NUM_CORES`), each with a local scratchpad,
  a **16×16 tensor-core MMA** engine, a command queue, and DMA,
- a **tile scheduler** that fans a large GEMM out into 16×16 blocks across the
  cores and accumulates over K-tiles,
- **7 dtypes** (fp32/fp16/bf16/i32/i16/i8/u8) with mixed precision, and
- a **CUDA/HIP-style runtime API**: alloc/upload/download, matmul, gemm
  (+bias +relu), matvec, elementwise, reduction, FMA, convert, plus streams,
  events, and barriers.

On top of the matrix engine it has a **vector / special-function unit** with the
operations a neural network actually needs:

| Group | Operations |
|-------|------------|
| Activations (24) | relu, leaky-relu, relu6, clamp, sigmoid, tanh, gelu (exact + tanh), silu/swish, elu, softplus, hardsigmoid, hardswish, mish, exp, log, sqrt, rsqrt, abs, neg, recip, sign, square, erf |
| Normalization | layer norm, RMS norm (with optional affine gamma/beta) |
| Attention/classifier | row-wise softmax and log-softmax (numerically stable) |
| Convolution | conv2d (lowered to im2col + tensor-core GEMM), im2col, max/avg pool2d |
| Tensor plumbing | transpose, gather (embedding lookup), dtype convert |
| Reductions | sum, max, min, mean, prod, argmax, argmin |
| Layers | `emunpu_linear`: GEMM + bias + any fused activation epilogue |

Every one of them fans out across the cores, is exercised by the test suite
against an independent host reference, and shows up in the trace and the live
visualizer. [`docs/MICRO_ISA.md`](docs/MICRO_ISA.md) is the normative
instruction-set spec.

**The units have real widths, and the ISA exposes them.** One instruction is one
issue of one functional unit, so it may only name as much work as that unit is
wide — 16×16×16 for the matrix engine, 256 elements for the vector ALU, 64 for the
special-function unit, one row (≤1024 wide) for the reduction unit, 64 KB for a
DMA descriptor. An instruction that asks for more is *rejected*, not silently
executed, which makes tiling an explicit obligation of whoever issues the
instructions. Query the limits with `emunpu_device_query_caps()`, and either let
the device API tile for you or do it yourself:

- **choose how many cores an operation uses** —
  `emunpu_device_set_exec_scope(dev, first_core, count)`, with
  `emunpu_device_core_stats()` to see what each core actually did;
- **program the cores directly** — `emunpu_device_local_alloc` /
  `emunpu_device_enqueue` / `emunpu_device_launch` let you write the instruction
  stream yourself: block the problem, assign tiles to cores, DMA operands into
  each core's scratchpad, issue MMA against local memory. That is what
  `demo/lowlevel_demo.c` does, and what a code generator targeting this device
  would emit.

Execution is functionally synchronous and bit-exact (no timing model): it exists
to validate the compiler↔runtime contract and to *see* work distribute across
cores, not to predict cycles.

## Layout

```
emunpu-sim/
├── include/emunpu/     Public headers — the API you compile against
│   ├── arch.h               architecture params, dtypes, and the per-core micro-ISA
│   ├── device.h             the device / runtime API (start here)
│   ├── global_mem.h         global memory allocator + tensor descriptors
│   ├── core.h               a single compute core (scratchpad + MMA engine)
│   ├── scheduler.h          GEMM → 16×16 tile schedule across cores
│   ├── viz.h                visualization event stream
│   └── export.h             EMUNPU_API visibility macro (shared-lib exports)
├── src/                Simulator implementation (7 .c files, IREE-independent)
│                       core.c = the kernels; ops.c = sub-op tables + activations
├── test/
│   ├── device_test.c        device/memory/scheduler plumbing (152 checks)
│   └── ops_test.c           ML ops, unit limits, core selection, raw
│                             instruction programming (531 checks)
├── demo/
│   ├── mlp_demo.c           2-layer MLP (C, no deps)
│   ├── cnn_demo.c           conv → relu → pool → conv → pool → linear → softmax
│   ├── transformer_demo.c   embedding → attention → layernorm → GELU FFN
│   ├── lowlevel_demo.c      hand-written instruction stream + core selection
│   └── visualizer.py        live terminal visualizer
├── docs/               MICRO_ISA.md (hardware ISA spec), ISA.md (executable
│                       format spec) + design deep-dives
├── reference/          IREE integration example (requires IREE to build) + ISA
├── Makefile            plain `make` build
└── CMakeLists.txt      CMake / add_subdirectory() build
```

Each demo validates itself against a double-precision host reference and exits
non-zero on mismatch, so the demos double as end-to-end tests.

## Build & run

Either build system works; both need only a C11 compiler and libm. The default
artifact is a **shared object** — `libemunpu.so` (`.dylib` on macOS,
`emunpu.dll` on Windows) — that exports exactly the documented device/runtime
API (built with hidden visibility; internal helpers stay private). A static
archive is also produced/available.

```bash
# Make
make            # shared object + static lib + tests + demos into ./build
make shared     # just libemunpu.so.1.0.0 (+ soname symlinks)
make test       # build & run every test and demo (linked to the .so)
make demo       # build & run just the demos

# or CMake (shared by default; -DEMUNPU_BUILD_SHARED=OFF for a static archive)
cmake -S . -B build && cmake --build build
ctest --test-dir build          # 6 tests: device_test, ops_test, + the 4 demos
./build/cnn_demo
cmake --install build --prefix /usr/local   # .so + headers + CMake package
```

The core count is a runtime knob: `IREE_EMUNPU_NUM_CORES=3 ctest --test-dir build`
re-runs everything on 3 cores (the tests use sizes that are not multiples of the
core count, so the ragged-edge splitting stays covered).

The exported ABI surface is exactly the API — confirm with
`nm -D --defined-only build/libemunpu.so.1 | grep ' T '` (all `emunpu_*`).

Expected demo output:

```
emunpu MLP demo:  x[64,128] -> relu(Linear 256) -> Linear 32
device: 10 cores
output y[0, 0:6] =  9.2772  4.8061 -3.3945 ...
max abs error vs host reference: 1.431e-06  -> PASS

emunpu CNN demo:  x[4,1,16,16]
  conv3x3 1->8 +relu | maxpool2 | conv3x3 8->16 +relu | avgpool2 | linear 256->10 | softmax
sample 0: class 3  p=0.7316   probs[0:5] = 0.2208 0.0001 0.0052 0.7316 0.0035
max abs error vs host reference: 2.292e-07
device argmax matches host: yes                          -> PASS

emunpu transformer demo:  24 tokens x 32 dims (ffn 64), vocab 64
attention weights: max abs error 1.439e-07, worst row-sum drift 3.089e-08
block output:      max abs error 2.430e-06                -> PASS

emunpu hardware capabilities
  cores 10 | global memory 256.0 MB | local scratchpad / core 256 KB
  per-instruction limits: MMA 16x16x16 | vector 256 | SFU 64 | row 1x<=1024 | DMA 64 KB
GEMM 72x56x40 -> 5 x 4 output tiles x 3 k-tiles = 60 MMA + 140 DMA instructions
enqueueing a 17x17x17 MMA (one past the engine's 16x16 tile): OUT_OF_RANGE
hand-scheduled GEMM: max abs error 0.000e+00  -> PASS
same GEMM through the high-level API, per execution scope:
   1 core : err 0.000e+00  instructions per core [200 0 0 0 0 0 0 0 0 0]
   4 cores: err 0.000e+00  instructions per core [50 50 50 50 0 0 0 0 0 0]
```

## Visualize a run

The engine emits a structured event stream when `IREE_EMUNPU_VIZ` names a file;
the Python visualizer tails it and animates per-core MMA / DMA / vector /
special-function activity. It's pure stdlib + ANSI, so it works over SSH. Use two
terminals:

```bash
# Terminal 1 — start the visualizer (owns the event file):
python demo/visualizer.py /tmp/emunpu.events

# Terminal 2 — run any workload pointing IREE_EMUNPU_VIZ at the same file:
IREE_EMUNPU_VIZ=/tmp/emunpu.events ./build/transformer_demo
```

`python demo/visualizer.py FILE --replay` re-plays a captured file once and
exits. Set `IREE_EMUNPU_TRACE=1` on any run to log every op the engine executes.
The MLP demo's matmuls are larger than 16×16, so work fans across all 10 cores;
tiny (≤16×16) matmuls run on a single tile (core 0 only). The CNN and transformer
demos are the most interesting to watch — they cycle through all four op classes
(`IM2COL`/`MMA16`/`POOL2D` and `NORM`/`MMA16`/`SOFTMAX`/`ACT.gelu`). See
[`docs/VISUALIZER.md`](docs/VISUALIZER.md).

## The API in 30 seconds

```c
#include "emunpu/device.h"

emunpu_device_t* dev;
emunpu_device_create(/*global_mem_bytes=*/0, &dev);   // 0 => default capacity

emunpu_tensor_id_t a, b, c;
uint32_t sa[2] = {M, K}, sb[2] = {N, K}, sc[2] = {M, N};   // B stored transposed
emunpu_device_alloc(dev, EMUNPU_DT_FP32, 2, sa, "A", &a);
emunpu_device_alloc(dev, EMUNPU_DT_FP32, 2, sb, "B", &b);
emunpu_device_alloc(dev, EMUNPU_DT_FP32, 2, sc, "C", &c);

emunpu_device_upload(dev, a, hostA, (uint64_t)M * K * sizeof(float));
emunpu_device_upload(dev, b, hostB, (uint64_t)N * K * sizeof(float));

// C = relu(A @ B^T + bias); NULL stream = execute immediately.
emunpu_gemm(dev, NULL, c, a, b, /*bias=*/0, M, N, K, /*transpose_b=*/1,
            /*fuse_relu=*/1);

emunpu_device_download(dev, c, hostC, (uint64_t)M * N * sizeof(float));
emunpu_device_destroy(dev);
```

The ML operations follow the same shape — one call per op, `NULL` stream to run
immediately, work tiled to the unit widths and fanned across the cores for you:

```c
// A linear layer with any fused activation epilogue.
emunpu_act_t gelu = {EMUNPU_ACT_GELU, /*alpha=*/0.0, /*beta=*/0.0};
emunpu_linear(dev, NULL, y, x, w, bias, M, N, K, /*transpose_b=*/1, &gelu);

emunpu_softmax(dev, NULL, p, y, /*rows=*/M, /*width=*/N, /*log_softmax=*/0);
emunpu_layernorm(dev, NULL, out, in, gamma, beta, rows, width, /*eps=*/1e-5);
emunpu_activation(dev, NULL, out, in, count, EMUNPU_ACT_LEAKY_RELU, 0.01, 0.0);
emunpu_reduce(dev, NULL, cls, p, M, N, EMUNPU_REDUCE_ARGMAX);   // -> i32 tensor

// conv2d: NCHW in, [out_channels, C*kh*kw] weight (a flattened torch weight).
emunpu_window_t win = {.batch = B, .channels = C, .in_h = H, .in_w = W,
                       .kernel_h = 3, .kernel_w = 3,
                       .stride_h = 1, .stride_w = 1, .pad_h = 1, .pad_w = 1};
emunpu_act_t relu = {EMUNPU_ACT_RELU, 0.0, 0.0};
emunpu_conv2d(dev, NULL, out, in, weight, bias, &win, out_channels, &relu);
emunpu_pool2d(dev, NULL, pooled, out, &pool_win, EMUNPU_POOL_MAX);
```

And the layer underneath, when you want the scheduling decisions yourself:

```c
emunpu_caps_t caps;
emunpu_device_query_caps(dev, &caps);          // what one instruction may do

// Run the next operations on 4 cores only.
emunpu_device_set_exec_scope(dev, /*first_core=*/0, /*core_count=*/4);
emunpu_matmul(dev, NULL, c, a, b, M, N, K, 1);
emunpu_device_reset_exec_scope(dev);

// ...or write the instruction stream yourself.
uint64_t tile;
emunpu_device_local_alloc(dev, core, caps.mma_dim * caps.mma_dim * 4, &tile);
emunpu_instr_t dma = {.op = EMUNPU_I_DMA_G2L, .a_dtype = EMUNPU_DT_FP32,
                      .c_dtype = EMUNPU_DT_FP32, .M = 16, .N = 16,
                      .a_offset = a_global, .a_row_stride = K * 4,
                      .c_offset = tile};
emunpu_device_enqueue(dev, core, &dma);        // rejected if it exceeds a unit
emunpu_device_launch(dev);

emunpu_core_stats_t st;
emunpu_device_core_stats(dev, core, &st);      // instrs / MMA issues / DMA bytes
```

The full surface (matvec, vector ops, reduce, fma, convert, transpose, im2col,
gather, streams/events, `emunpu_matmul_scheduled` for replaying a
compiler-decided tile schedule) is documented inline in
[`include/emunpu/device.h`](include/emunpu/device.h);
[`docs/MICRO_ISA.md` §8](docs/MICRO_ISA.md#8-how-the-device-api-lowers-onto-these-instructions)
shows which instructions each call issues.

## Integrating into your compiler stack

Two layers, cleanly separated:

1. **The hardware model** (`include/` + `src/`) is what your *runtime* drives.
   Link `libemunpu` (the shared object, or the static archive) and call the
   device API, exactly as your HAL/runtime would call a CUDA/HIP driver. This is
   100% portable C. Installed projects can `find_package`/link the
   `emunpu::emunpu` CMake target, or just `-lemunpu` against the installed
   headers + `.so`. Every public function is tagged `EMUNPU_API` (see
   `include/emunpu/export.h`); define `EMUNPU_STATIC` when linking the archive.
   Its instruction set is specified in [`docs/MICRO_ISA.md`](docs/MICRO_ISA.md),
   which also has a checklist ([§13](docs/MICRO_ISA.md#13-adding-an-opcode)) for
   adding an opcode of your own.

2. **The compiler↔runtime contract** is `reference/emunpu_isa.h`, formally
   specified in [`docs/ISA.md`](docs/ISA.md): a compact command/tile-schedule
   format your *code generator* emits and the runtime decodes. `reference/` also
   ships the original IREE integration verbatim
   (`emunpu_hw.c`, `emunpu_loader.c`, the FlatBuffer memory layout) as a worked
   example of wiring the simulator behind a real HAL — **these files require
   IREE to compile** and are provided as a reference, not built by this project.

   Note the deliberate asymmetry: that blob format (v4) exposes only the four
   coarsest ops (`MATMUL`/`ADD`/`RELU`/`COPY`), so a compiler targeting it reaches
   the tensor cores but not the activation/softmax/conv units. Everything the
   hardware can do is reachable from layer 1 today;
   [`docs/ISA.md` Appendix C](docs/ISA.md#appendix-c--non-normative-proposed-v5-opcodes)
   records what a v5 format would need to expose the rest to a code generator.

See [`reference/INTEGRATION.md`](reference/INTEGRATION.md) for how the pieces fit
and where to hook your own backend, [`docs/MICRO_ISA.md`](docs/MICRO_ISA.md) for
the instruction set, and [`docs/MULTICORE.md`](docs/MULTICORE.md) /
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the architecture deep-dives
(note: file paths in the deep-dive docs refer to the original IREE source tree).

## License

Apache-2.0 WITH LLVM-exception. See [`LICENSE`](LICENSE).
