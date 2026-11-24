// Round-trip smoke test for the tldr dialect: run through tldr-opt and expect
// it to parse, verify, and print back unchanged (mlir-opt's usual contract).
//
// matmul is real linalg.matmul, not a tldr op -- it already means exactly
// what we want, so there's no reason to reinvent it. relu/softmax stay as
// tldr ops because nothing in MLIR represents "one atomic LAUNCH kernel
// call" the way the runtime needs it.

func.func @smoke(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %init = tensor.empty() : tensor<4x16xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
                       outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %r = tldr.relu %mm : tensor<4x16xf32>
  %s = tldr.softmax %r {log_softmax = false} : tensor<4x16xf32>
  return %s : tensor<4x16xf32>
}
