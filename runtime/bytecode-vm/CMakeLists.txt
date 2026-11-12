# ============================================================================
# CMake build for the bytecode VM runtime.
#
# WHAT THIS BUILDS (by default, emunpu only -- no GPU needed):
#   - vm_core_lib   : the VM itself (parser, core, io, emunpu device), compiled
#                     once and shared by every executable below.
#   - emunpu_test   : runs one hard-coded program (src/main_emunpu.cpp).
#   - test_runner   : runs any bytecode file by name (src/test_runner.cpp),
#                     used by the test folders.
#
# OPTIONAL (only when you have a CUDA GPU):
#   - vm_cuda       : the CUDA entry point + kernels. Off by default.
#                     Turn on with:  -DVM_ENABLE_CUDA=ON
#
# HOW TO BUILD (run these from this folder, runtime/bytecode-vm/):
#   cmake -S . -B build          # 1. generate the build files into build/
#   cmake --build build          # 2. compile everything
#
# The compiled programs land in build/. Run one like:
#   cd src/test-bytecode && ../../build/test_runner tests/ok_minimal.txt
#
# TO ADD A NEW .cpp FILE to the VM: add its path to the VM_SOURCES list below.
# TO ADD A NEW EXECUTABLE: copy an add_executable(...) + target_link_libraries(...)
# pair near the bottom and change the names.
# ============================================================================

cmake_minimum_required(VERSION 3.14)
project(bytecode_vm LANGUAGES CXX)

# Use C++14 (the clang++ command we used asked for -std=c++14). REQUIRED means
# "fail loudly if the compiler can't do it", instead of silently falling back.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# emunpu-sim: build it as part of our build, straight from its own CMakeLists.
#
# add_subdirectory() pulls in emunpu-sim's build recipe, so we never depend on a
# pre-built libemunpu.a sitting in the right place -- CMake compiles it for us
# and gives us its target, `emunpu::emunpu`. The path is relative to THIS file:
# runtime/bytecode-vm/ -> ../../emunpu-sim.
#
# The second argument is just a name for where its build output goes inside our
# build/ folder.
# ----------------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../emunpu-sim emunpu-sim-build)

# ----------------------------------------------------------------------------
# The VM itself, compiled ONCE into a static library.
#
# All three executables need these same files. Compiling them once here and
# linking the result into each executable avoids listing them three times (and
# avoids compiling them three times).
# ----------------------------------------------------------------------------
set(VM_SOURCES
    src/bytecode.cpp
    src/verifier.cpp
    src/vm_core.cpp
    src/io_handle.cpp
    src/hal/emunpu_device.cpp
)

add_library(vm_core_lib STATIC ${VM_SOURCES})

# Header search paths. PUBLIC means "anything that links vm_core_lib also gets
# these include paths" -- so the executables below don't have to repeat them.
target_include_directories(vm_core_lib PUBLIC
    src
    src/hal
)

# Link emunpu into the VM library. Because this is PUBLIC, the executables that
# link vm_core_lib get emunpu linked transitively -- no need to repeat it.
target_link_libraries(vm_core_lib PUBLIC emunpu::emunpu)

# ----------------------------------------------------------------------------
# Executables. Each is one file + the shared VM library.
# ----------------------------------------------------------------------------
add_executable(emunpu_test src/main_emunpu.cpp)
target_link_libraries(emunpu_test PRIVATE vm_core_lib)

add_executable(test_runner src/test_runner.cpp)
target_link_libraries(test_runner PRIVATE vm_core_lib)

# Host-API and kernel-numeric tests that bytecode files cannot reach.
# Run with: ./build/unit_tests
add_executable(unit_tests src/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE vm_core_lib)

# ----------------------------------------------------------------------------
# CUDA backend -- OFF by default. This Mac has no CUDA toolchain, so we never
# even look for it unless the flag is on. Turn it on with:
#   cmake -S . -B build -DVM_ENABLE_CUDA=ON
# ----------------------------------------------------------------------------
option(VM_ENABLE_CUDA "Build the CUDA backend and its entry point" OFF)

if(VM_ENABLE_CUDA)
    # Tell CMake this project now also uses the CUDA language. Done inside the
    # if() so a machine without CUDA never triggers the check.
    enable_language(CUDA)

    add_executable(cuda_test
        src/main.cpp
        src/hal/cuda_device.cpp
        src/hal/kernels.cu
    )
    target_link_libraries(cuda_test PRIVATE vm_core_lib)

    message(STATUS "CUDA backend: ENABLED")
else()
    message(STATUS "CUDA backend: disabled (use -DVM_ENABLE_CUDA=ON to build it)")
endif()
