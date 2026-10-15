

import torch
import torch.nn as nn
import numpy as np
import iree.turbine.aot as aot
import iree.runtime as ireert


# class MatmulModel(nn.Module):
#     def forward(self, A, B):
#         return A @ B


# class ReluMatrixModel(nn.Module):
#     def __init__(self):
#         super(ReluMatrixModel, self).__init__()
#         # Initialize the ReLU activation layer
#         self.relu = nn.ReLU()

#     def forward(self, matrix_a, matrix_b):
#         # 1. Element-wise addition
#         added_matrix = matrix_a + matrix_b
        
#         # 2. Apply ReLU (keeps positive values, changes negative ones to 0)
#         output_matrix = self.relu(added_matrix)
#         return output_matrix


class SimpleMSEModel(nn.Module):
    def forward(self, A, B):
        # Mean Squared Error between A and B directly: 2 inputs -> 1 scalar output.
        return torch.mean((A - B) ** 2)


def compile_and_verify(model, example_args, function_name, mlir_path, vmfb_path):
    """Runs the full IREE pipeline for any model: export -> compile -> load ->
    invoke -> verify against the real PyTorch output. Reused across every
    sample model (matmul, MSE, MLP, ...) so the pipeline is written once.
    """
    # 1. PyTorch -> MLIR
    export_output = aot.export(
        model, args=example_args, module_name="module", function_name=function_name
    )
    export_output.save_mlir(mlir_path)
    print(f"[1/4] Exported MLIR -> {mlir_path}")

    # 2. MLIR -> compiled .vmfb (AOT compilation, target = local CPU
    #    since Brief 4 tests against a mock/stub, not a real device backend)
    export_output.compile(save_to=vmfb_path)
    print(f"[2/4] Compiled -> {vmfb_path}")

    # 3. Load the compiled module with IREE's high-level runtime
    config = ireert.Config("local-task")  # run on only CPU
    ctx = ireert.SystemContext(config=config)  # runtime session

    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()

    vm_module = ireert.VmModule.copy_buffer(config.vm_instance, vmfb_data)
    ctx.add_vm_module(vm_module)
    print("[3/4] Loaded compiled module into runtime session")

    # 4. Invoke the entry function and read outputs (high-level call shape:
    #    load_module -> call(entry) -> pass inputs -> invoke -> get_output)
    entry = ctx.modules.module[function_name]
    numpy_args = [arg.numpy() for arg in example_args]
    result = entry(*numpy_args)
    print("[4/4] Invoked entry function, got result")

    # Reference check against the PyTorch model itself
    expected = model(*example_args).detach().numpy()
    result_np = np.asarray(result)
    max_err = float(np.max(np.abs(result_np - expected)))
    print(f"Max abs error vs PyTorch reference: {max_err:.3e}")
    assert np.allclose(result_np, expected, atol=1e-4), "Mismatch vs PyTorch reference!"
    print("MATCH: IREE-executed result equals PyTorch reference within tolerance.")
    return result_np


def main():
    model = SimpleMSEModel()
    # Matches orchestrator.cpp's Modelsignature: 2 inputs of shape [2,2], scalar output.
    M, N = 2, 2
    example_args = (torch.randn(M, N), torch.randn(M, N))
    compile_and_verify(
        model, example_args, "simple_mul", "brief4/mse.mlir", "brief4/mse.vmfb"
    )


if __name__ == "__main__":
    main()
