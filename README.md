# MLIR DSL Compiler

A compiler for a small DSL of AI and ML operators. It lowers the DSL through MLIR into bytecode, and a runtime runs that bytecode on an emulated NPU (`emunpu-sim`). The target is a downsized Llama-3 model.

## Status

| Part                       | Folder                  | Status                                                      |
| -------------------------- | ----------------------- | ----------------------------------------------------------- |
| DSL frontend               | `Brief-1 DSL Frontend/` | Working: parse, AST, semantic analysis, JSON and DOT output |
| MLIR dialects and lowering | —                       | Not in this repository yet                                  |
| Runtime (bytecode VM)      | `runtime/bytecode-vm/`  | Working on emunpu. CUDA backend not verified.               |
| Emulated NPU               | `emunpu-sim/`           | Given library, not changed here                             |

## Pipeline

```
DSL source (.tldr) -> MLIR dialects -> bytecode -> runtime VM -> emunpu-sim
```

## Quick start

Build and run the DSL frontend:

```
cd "Brief-1 DSL Frontend"
cmake --preset default
cmake --build --preset default
./build/tldr_compiler tests/test_matmul.tldr --json
```

Build and test the runtime:

```
cd runtime/bytecode-vm
cmake -S . -B build
cmake --build build
./build/test_runner src/test-bytecode/tests/ok_launch_relu.txt
./build/unit_tests
```

## Repository layout

```
Brief-1 DSL Frontend/   DSL compiler front end (C++17, CMake)
emunpu-sim/             Emulated multi-core NPU (C11)
runtime/                Bytecode VM, device backends, design notes
```

## Documentation

- [Bytecode contract](runtime/doc/bytecode-contract.md): the format the compiler must emit
- [Runtime design](runtime/doc/bytecode-vm-project.md): how the VM works
- [Validation design](runtime/bytecode-vm/validation-design.md): where each check lives
- [Device interface](runtime/bytecode-vm/hal-design.md): the `Device` API
- [DSL syntax](Brief-1%20DSL%20Frontend/DSL_Syntax.md) and [grammar](Brief-1%20DSL%20Frontend/grammar.md)
- [emunpu-sim](emunpu-sim/README.md) and its [instruction set](emunpu-sim/docs/MICRO_ISA.md)
