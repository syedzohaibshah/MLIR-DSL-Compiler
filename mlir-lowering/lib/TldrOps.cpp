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
