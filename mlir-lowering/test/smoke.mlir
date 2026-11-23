// Round-trip smoke test for the tldr dialect: run through tldr-opt and expect
// it to parse, verify, and print back unchanged (mlir-opt's usual contract).

func.func @smoke(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %g: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %mm = tldr.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = tldr.relu %mm : tensor<4x16xf32>
  %s = tldr.softmax %r {log_softmax = false} : tensor<4x16xf32>
  return %s : tensor<4x16xf32>
}
