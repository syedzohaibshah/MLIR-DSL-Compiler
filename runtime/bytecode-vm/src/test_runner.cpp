// Small test runner for the error-handling work.
//
// It takes ONE bytecode filename on the command line, loads it, sets a few
// generic input slots, runs the VM on the emunpu device, and prints the final
// Error. It does NOT check a numeric result -- each test file targets one or
// two edge cases, so the thing under test is the Error (or ok) that comes back.
//
// Build (from runtime/bytecode-vm):
//   clang++ -std=c++14 -I src -I src/hal -I ../../emunpu-sim/include \
//     src/bytecode.cpp src/io_handle.cpp src/vm_core.cpp \
//     src/hal/emunpu_device.cpp src/test_runner.cpp \
//     -L ../../emunpu-sim/build -lemunpu -lm -o test_runner
//
// Run one test (from the folder that holds the .txt file):
//   ./test_runner tests/err_bad_number.txt
//
// Output line to read: "RESULT: ok" or "RESULT: FAIL - <message> (line N)".
#include <iostream>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"
#include "error.h"

// Runs one bytecode file. Returns the first Error, or ok().
static Error runFile(const char* path){
    Bytecode bytecode;
    RETURN_IF_ERROR(bytecode.loadFromFile(path));

    emunpuDevice npu;
    RETURN_IF_ERROR(npu.status());
    VM vm(&npu);

    // Generic inputs. Slots 0..4, 16 bytes each (four floats). A test that needs
    // a COPY_H2D on one of these slots gets a valid, consistent-size source.
    // A test about a MISSING input simply uses a slot we do not set here.
    static float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for(int slot = 0; slot < 5; ++slot){
        RETURN_IF_ERROR(vm.setInput(slot, a, sizeof(a)));
    }

    RETURN_IF_ERROR(vm.execute(bytecode.get_instructions()));
    return ok();
}

int main(int argc, char** argv){
    if(argc != 2){
        std::cerr << "usage: " << argv[0] << " <bytecode-file>" << std::endl;
        return 2;
    }

    Error e = runFile(argv[1]);
    if(!e.ok){
        std::cout << "RESULT: FAIL - " << e.msg << " (line " << e.line_number << ")" << std::endl;
        return 1;
    }
    std::cout << "RESULT: ok" << std::endl;
    return 0;
}
