add_executable(tldr-import tldr-import.cpp)

target_link_libraries(tldr-import PRIVATE
  MLIRTldr
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
