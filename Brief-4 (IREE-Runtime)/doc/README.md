# Brief 4 — documentation index

Brief 4 started as an IREE runtime orchestrator, then pivoted to a from-scratch bytecode VM after a team discussion. Both phases are documented here.

## Docs

- **[bytecode-vm-project.md](./bytecode-vm-project.md)** — the current work. Bytecode format, architecture, how the VM turns a text file into real device execution against either of two hardware backends, and where it's headed next.
- **[hal-design.md](../bytecode-vm/hal-design.md)** — the design document for the multi-hardware abstraction bytecode-vm-project.md describes: why bytecode is target-specific, the `Device` interface's contract, how `CudaDevice`/`EmunpuDevice` fit it, and what's still open.
- **[iree-runtime.md](./iree-runtime.md)** — how IREE's own runtime executes a compiled `.vmfb`, phase by phase (FlatBuffer ingestion, VM bytecode interpreter, HAL command recording). 

- **[Notes.md](./Notes.md)** — Brief 4's original scope (the IREE orchestrator phase): what the brief asked for, what IREE is, its architecture, and the pipeline stage by stage.

## Diagrams (Excalidraw)



- **[vm-architecture.excalidraw.svg](../vm-architecture.excalidraw.svg)** — the bytecode VM's own architecture, referenced from [bytecode-vm-project.md](./bytecode-vm-project.md#41-overview)
- **[IREE.excalidraw.svg](../IREE.excalidraw.svg)** — IREE runtime notes
- **[LLVM.excalidraw.svg](../LLVM.excalidraw.svg)** — LLVM infrastructure notes
- **[MLIR.excalidraw.svg](../MLIR.excalidraw.svg)** — MLIR compiler notes
- **[llama3.excalidraw.svg](../llama3.excalidraw.svg)** — Llama3 model architecture
