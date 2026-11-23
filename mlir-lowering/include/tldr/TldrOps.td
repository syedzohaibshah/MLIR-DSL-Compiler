#ifndef TLDR_OPS
#define TLDR_OPS

include "tldr/TldrDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonTypeConstraints.td"

// LAUNCH op_id 2 (RELU). count is the input's element count at lowering
// time; there is no scalar for it here, it's recovered from the tensor type.
def Tldr_ReluOp : Tldr_Op<"relu", [Pure, AllTypesMatch<["input", "result"]>]> {
  let summary = "Elementwise ReLU. Maps to LAUNCH op_id 2 (RELU).";
  let arguments = (ins RankedTensorOf<[F32]>:$input);
  let results = (outs RankedTensorOf<[F32]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// LAUNCH op_id 5 (SOFTMAX). The runtime's SOFTMAX kernel is row-wise over a
// 2D buffer only (rows, width) — it has no generic axis, so this op is
// restricted to rank-2 tensors on purpose, not as a temporary limitation.
def Tldr_SoftmaxOp : Tldr_Op<"softmax", [Pure, AllTypesMatch<["input", "result"]>]> {
  let summary = "Row-wise softmax over a rank-2 tensor. Maps to LAUNCH op_id 5 (SOFTMAX).";
  let arguments = (ins
    RankedTensorOf<[F32]>:$input,
    DefaultValuedAttr<BoolAttr, "false">:$log_softmax
  );
  let results = (outs RankedTensorOf<[F32]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

// LAUNCH op_id 0 (MATMUL). result = lhs @ rhs, all rank-2, K dims must agree.
def Tldr_MatMulOp : Tldr_Op<"matmul", [Pure]> {
  let summary = "C = A @ B. Maps to LAUNCH op_id 0 (MATMUL).";
  let arguments = (ins
    RankedTensorOf<[F32]>:$lhs,
    RankedTensorOf<[F32]>:$rhs
  );
  let results = (outs RankedTensorOf<[F32]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // TLDR_OPS
