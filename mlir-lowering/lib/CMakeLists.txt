add_mlir_dialect_library(MLIRTldr
  TldrDialect.cpp
  TldrOps.cpp

  DEPENDS
  MLIRTldrOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
)
