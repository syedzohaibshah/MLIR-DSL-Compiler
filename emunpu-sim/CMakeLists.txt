# Standalone CMake build for the emulated multi-core NPU simulator.
#
#   cmake -S . -B build && cmake --build build
#   ctest --test-dir build            # run the functional test
#   ./build/mlp_demo                  # run the demo
#   cmake --install build --prefix /where/you/want   # install .so + headers
#
# Builds a SHARED library (libemunpu.so / .dylib / emunpu.dll) by default; only
# the EMUNPU_API-annotated device/runtime API is exported. Set
# -DEMUNPU_BUILD_SHARED=OFF for a static archive instead. Drop it into your own
# project with add_subdirectory() and link the `emunpu` target either way.

cmake_minimum_required(VERSION 3.16)
project(emunpu_sim VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(EMUNPU_BUILD_SHARED "Build emunpu as a shared library" ON)
option(EMUNPU_BUILD_TESTS "Build the functional test" ON)
option(EMUNPU_BUILD_DEMO "Build the end-to-end MLP demo" ON)

include(GNUInstallDirs)

# --- The simulator library --------------------------------------------------
file(GLOB EMUNPU_SRCS CONFIGURE_DEPENDS src/*.c)

if(EMUNPU_BUILD_SHARED)
  add_library(emunpu SHARED ${EMUNPU_SRCS})
  # Export only EMUNPU_API symbols; keep every internal helper private.
  set_target_properties(emunpu PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  target_compile_definitions(emunpu PRIVATE EMUNPU_BUILDING_SHARED)
else()
  add_library(emunpu STATIC ${EMUNPU_SRCS})
  # Consumers of the static lib must strip the import/export decoration.
  target_compile_definitions(emunpu PUBLIC EMUNPU_STATIC)
endif()

add_library(emunpu::emunpu ALIAS emunpu)

target_include_directories(emunpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(NOT MSVC)
  target_compile_options(emunpu PRIVATE -Wall -Wextra)
  target_link_libraries(emunpu PUBLIC m)
endif()

# --- Functional tests -------------------------------------------------------
# device_test covers the device/memory/scheduler plumbing; ops_test covers the
# ML operation set (activations, softmax, norms, conv, pooling, gather, ...).
if(EMUNPU_BUILD_TESTS)
  enable_testing()
  foreach(_test device_test ops_test)
    add_executable(${_test} test/${_test}.c)
    target_link_libraries(${_test} PRIVATE emunpu)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()
endif()

# --- End-to-end demos -------------------------------------------------------
# Each demo validates itself against a host reference and exits non-zero on
# mismatch, so they double as end-to-end tests.
if(EMUNPU_BUILD_DEMO)
  foreach(_demo mlp_demo cnn_demo transformer_demo lowlevel_demo)
    add_executable(${_demo} demo/${_demo}.c)
    target_link_libraries(${_demo} PRIVATE emunpu)
    if(EMUNPU_BUILD_TESTS)
      add_test(NAME ${_demo} COMMAND ${_demo})
    endif()
  endforeach()
endif()

# --- Install ----------------------------------------------------------------
# Installs the library and the public headers so downstream projects can
# find_package(emunpu) / link emunpu::emunpu against an installed .so.
install(TARGETS emunpu
  EXPORT emunpuTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/emunpu DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT emunpuTargets
  FILE emunpuTargets.cmake
  NAMESPACE emunpu::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/emunpu)
