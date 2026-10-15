Once the compile-time pipeline generates a Virtual Machine FlatBuffer (`.vmfb`), the **IREE Runtime** executes it through a highly optimised, low-overhead pipeline designed to bypass dynamic memory allocations, minimise host-side latency, and schedule asynchronous tasks directly on target devices. 

The runtime pipeline processes the loaded `.vmfb` through the following sequence:

---

### Phase 1: FlatBuffer Ingestion & Module Registration
The pipeline begins when the host application loads the compiled `.vmfb` into memory (often via zero-copy memory mapping):

1. **Instance and Context Allocation:** The runtime initializes the core virtual machine engine by creating a `iree_vm_instance_t` and a `iree_vm_context_t`.
2. **Module Ingestion:** The `.vmfb` payload is instantiated as a bytecode module (`iree_vm_module_t`). The host application then registers this compiled bytecode alongside system modules—most importantly, the **Hardware Abstraction Layer (HAL) Module** (`iree_hal_module`)—using `iree_vm_context_register_modules`.
3. **State Resolution:** Upon context creation, the virtual machine resolves dependencies and allocates the module state. This resolves static data and registers read-only constant tables (`rodata_buffers`) in the heap-state.

---

### Phase 2: The Virtual Machine Bytecode Interpreter
When a model function is invoked (e.g. via `iree_vm_invoke`), execution enters the VM Interpreter. To maintain high performance and deterministic execution, this interpreter operates under strict rules:

1. **Static Allocations via C lists:** To avoid runtime heap allocation overheads during parameter passing, inputs and outputs are managed through `iree_vm_list_t`. The static backing storage size is calculated upfront using `iree_vm_list_storage_size` to allow stack-allocated parameters.
2. **Controlled Reference-Counting:** As bytecode runs, the VM manages reference-counted objects (such as lists or buffers) using explicit C API controls like `iree_vm_ref_retain_or_move_checked`.
3. **Bytecode Dispatch Loop:** The interpreter dispatches the low-level instructions. Control flow is evaluated sequentially. For example, conditional branches (`CondBranchOpConversion`) are translated into flat instruction jumps.
4. **Boundary Calls to the HAL Module:** When the interpreter hits a compiled hardware boundary, it makes a lightweight call to the registered **HAL Module**, handing off bytecode execution to the target device driver.

---

### Phase 3: Hardware Abstraction Layer (HAL) Command recording
The HAL driver translates logical operations into real hardware executions. Rather than executing kernels immediately, the runtime records them into reusable command buffers (`iree_hal_command_buffer_t`):

1. **Resource Mapping:** VM reference arguments are mapped to physical hardware buffers (`iree_hal_buffer_t`) and timeline semaphores (`iree_hal_semaphore_t`).
2. **Buffer Metadata Management:** To achieve zero-copy performance, the allocator caches resources, utilizing static buffer registries and rodata segment mappings.
3. **Kernel Argument (`kernarg`) templates:** For massive execution paths (like AMDGPU/HIP), recording the exact arguments for each dispatch can be a major host-side bottleneck. IREE bypasses this by constructing **Prepublished or Patched Kernarg templates**. Static kernel arguments are baked once during compilation or finalizing, while dynamic parameters (like changing offsets) are patched directly into the template at replay time.
4. **Command Appending:** The HAL driver appends GPU commands into the recording arena. Dispatches are structured with planned layouts and binding sources. Execution barriers with system/agent-scope memory-coherency are accumulated and set.
5. **Finalization:** Once recording completes, the command buffer resource-set is frozen. Reusable command buffers are cached to avoid recording costs on subsequent iterations.

---

### Phase 4: Asynchronous Queue Execution & Proactor Scheduling
Finalized command buffers are submitted to the target device's queue. Synchronization across CPU, GPU, and network queues is managed asynchronously on a unified timeline:

1. **Timeline Semaphores:** Execution queues do not block the host CPU. Instead, tasks are chained using asynchronous timeline semaphores. The host CPU registers wait tasks and yields immediately.
2. **Platform-Native Event Loops:** To resolve wait queues with zero overhead, the async engine implements proactor event loops tailored to specific host operating systems (e.g. Windows IOCP or Linux `io_uring`):
   * **On Windows:** It leverages native wait packets (`NtCreateWaitCompletionPacket`) and Futex-style address waits (`WaitOnAddress`).
   * **On Linux:** It leverages `io_uring` to bridge device wakes directly to the poll loop.
3. **Zero-Copy Memory Pipelines:** Large compute operations bypass host staging altogether. Zero-copy read/write capabilities are supported by registering slab regions and buffer registration states directly with the OS proactor.
4. **Execution and Output Export:** Once the device completes the queue task, the timeline semaphore signals completion. The resulting output is mapped back or exported directly as a finished buffer view (`iree_hal_buffer_view_t`) back to the host application.

***

