add_llvm_executable(tldr-opt tldr-opt.cpp)
llvm_update_compile_flags(tldr-opt)

target_link_libraries(tldr-opt PRIVATE
  MLIRTldr
  MLIROptLib
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
)

mlir_check_all_link_libraries(tldr-opt)
