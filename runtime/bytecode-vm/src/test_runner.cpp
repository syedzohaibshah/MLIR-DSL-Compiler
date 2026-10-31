// Small test runner for the runtime (parser + VM + emunpu device).
//
// It takes ONE bytecode filename, loads it, sets generic input slots, runs the
// VM, and prints a clear PASS / FAIL line based on a naming convention:
//   - a file named  err_*  MUST fail  (the VM must return an Error)
//   - a file named  ok_*   MUST pass  (the VM must return ok)
// Any other name is run and its raw result is printed, with no verdict.
//
// The runner decides the expected outcome from the FILE NAME, so one binary
// checks every test with no per-test C++.
//

// Run one test:
//   ./test_runner src/test-bytecode/tests/err_bad_number.txt
//

// Exit code: 0 = verdict PASS, 1 = verdict FAIL, 2 = usage / cannot open.
#include <iostream>
#include <string>
#include <cstring>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"
#include "error.h"

// Runs one bytecode file on the emunpu device. Returns the first Error, or ok().
// Device log lines (the "[emunpu] ..." prints) are not silenced here; the
// run script sends them to a detail log so the PASS/FAIL summary stays clean.
static Error runFile(const char* path){
    Bytecode bytecode;
    RETURN_IF_ERROR(bytecode.loadFromFile(path));

    emunpuDevice npu;
    RETURN_IF_ERROR(npu.status());
    VM vm(&npu);

    // Generic inputs: slots 0..7, 16 bytes each (four floats). A COPY_H2D on any
    // of these slots with size 16 gets a valid, consistent source. A test about
    // a MISSING input simply targets a slot above 7.
    static float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for(int slot = 0; slot < 8; ++slot){
        RETURN_IF_ERROR(vm.setInput(slot, a, sizeof(a)));
    }

    RETURN_IF_ERROR(vm.execute(bytecode.get_instructions()));
    return ok();
}

// Returns the base file name (after the last '/'), so the verdict reads cleanly.
static std::string baseName(const std::string& path){
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

int main(int argc, char** argv){
    if(argc != 2){
        std::cerr << "usage: " << argv[0] << " <bytecode-file>" << std::endl;
        return 2;
    }

    std::string name = baseName(argv[1]);
    Error e = runFile(argv[1]);

    // What outcome does the name promise?
    bool expect_fail = name.rfind("err_", 0) == 0;
    bool expect_ok   = name.rfind("ok_", 0) == 0;

    // One readable outcome string for the detail column.
    std::string outcome = e.ok
        ? std::string("ok")
        : std::string("ERROR: ") + e.msg + " (line " + std::to_string(e.line_number) + ")";

    // Verdict against the naming convention.
    if(expect_fail){
        if(!e.ok){ std::cout << "PASS  " << name << "  ->  " << outcome << std::endl; return 0; }
        std::cout << "FAIL  " << name << "  ->  expected an error, got ok" << std::endl; return 1;
    }
    if(expect_ok){
        if(e.ok){ std::cout << "PASS  " << name << "  ->  ok" << std::endl; return 0; }
        std::cout << "FAIL  " << name << "  ->  expected ok, got " << outcome << std::endl; return 1;
    }

    // No convention in the name: just report, no verdict.
    std::cout << "----  " << name << "  ->  " << outcome << std::endl;
    return e.ok ? 0 : 1;
}
