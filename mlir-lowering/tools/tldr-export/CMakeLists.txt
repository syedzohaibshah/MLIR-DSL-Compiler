add_llvm_executable(tldr-export tldr-export.cpp)
llvm_update_compile_flags(tldr-export)

target_link_libraries(tldr-export PRIVATE
  MLIRTldr
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(tldr-export)
