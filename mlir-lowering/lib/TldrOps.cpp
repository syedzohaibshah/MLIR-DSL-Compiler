#include "tldr/TldrOps.h"

#define GET_OP_CLASSES
#include "tldr/TldrOps.cpp.inc"

using namespace tldr;
using namespace mlir;

LogicalResult SoftmaxOp::verify() {
  auto type = llvm::cast<RankedTensorType>(getInput().getType());
  if (type.getRank() != 2)
    return emitOpError("expects a rank-2 tensor (rows x width); got rank ")
           << type.getRank();
  return success();
}

LogicalResult MatMulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return emitOpError("expects lhs, rhs and result to be rank-2 tensors");

  int64_t kLhs = lhsType.getDimSize(1);
  int64_t kRhs = rhsType.getDimSize(0);
  if (!ShapedType::isDynamic(kLhs) && !ShapedType::isDynamic(kRhs) &&
      kLhs != kRhs)
    return emitOpError("lhs's K dimension (")
           << kLhs << ") does not match rhs's K dimension (" << kRhs << ")";

  int64_t mLhs = lhsType.getDimSize(0);
  int64_t mResult = resultType.getDimSize(0);
  if (!ShapedType::isDynamic(mLhs) && !ShapedType::isDynamic(mResult) &&
      mLhs != mResult)
    return emitOpError("result's M dimension (")
           << mResult << ") does not match lhs's M dimension (" << mLhs
           << ")";

  int64_t nRhs = rhsType.getDimSize(1);
  int64_t nResult = resultType.getDimSize(1);
  if (!ShapedType::isDynamic(nRhs) && !ShapedType::isDynamic(nResult) &&
      nRhs != nResult)
    return emitOpError("result's N dimension (")
           << nResult << ") does not match rhs's N dimension (" << nRhs
           << ")";

  return success();
}
