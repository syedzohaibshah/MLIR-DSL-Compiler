#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "tldr/TldrDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<tldr::TldrDialect, mlir::func::FuncDialect,
                   mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tldr-opt\n", registry));
}
