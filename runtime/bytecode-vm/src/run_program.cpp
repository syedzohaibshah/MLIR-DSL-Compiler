// Generic bytecode-contract.md runner for compiler-emitted programs.
//
// test_runner.cpp and main_emunpu.cpp both bake in a fixed input/output
// signature (16-byte-only inputs, or a hardcoded 5-slot 2-layer network).
// A compiler-generated program declares its own signature in its header, so
// this runner reads that signature back and synthesizes input data to fit
// it, instead of assuming any particular slot layout. That's what lets one
// binary run any bytecode file the compiler emits, not just one fixed demo.
//
// Input data is deterministic, not random: for an N-float input slot, slot
// i gets the value (i - N/2), which is negative for the first half and
// non-negative for the second -- enough to exercise a RELU-style kernel
// meaningfully (some elements clipped to 0, some passed through unchanged).
#include <iostream>
#include <vector>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"
#include "error.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <bytecode-file>" << std::endl;
        return 2;
    }

    Bytecode bytecode;
    Error err = bytecode.loadFromFile(argv[1]);
    if (!err.ok) { report(err); return 1; }

    emunpuDevice npu;
    err = npu.status();
    if (!err.ok) { report(err); return 1; }

    VM vm(&npu);
    err = vm.bindSignature(bytecode);
    if (!err.ok) { report(err); return 1; }

    // Kept alive until execute() runs -- setInput only remembers a pointer.
    std::vector<std::vector<float>> inputBuffers;

    for (std::map<int, SlotDecl>::const_iterator it = bytecode.inputs().begin();
         it != bytecode.inputs().end(); ++it) {
        int slot = it->first;
        const SlotDecl& decl = it->second;
        if (decl.bytes % 4 != 0) {
            std::cerr << "run_program only generates float input data; slot "
                      << slot << " is " << decl.bytes << " bytes" << std::endl;
            return 1;
        }
        int count = decl.bytes / 4;
        std::vector<float> data(count);
        for (int i = 0; i < count; ++i)
            data[i] = static_cast<float>(i) - static_cast<float>(count) / 2.0f;
        inputBuffers.push_back(data);

        err = vm.setInput(slot, inputBuffers.back().data(), decl.bytes);
        if (!err.ok) { report(err); return 1; }

        std::cout << "input  slot " << slot << " (" << decl.name << "):";
        for (size_t i = 0; i < inputBuffers.back().size(); ++i)
            std::cout << " " << inputBuffers.back()[i];
        std::cout << std::endl;
    }

    err = vm.execute(bytecode.get_instructions());
    if (!err.ok) { report(err); return 1; }

    for (std::map<int, SlotDecl>::const_iterator it = bytecode.outputs().begin();
         it != bytecode.outputs().end(); ++it) {
        int slot = it->first;
        const SlotDecl& decl = it->second;
        std::vector<float> out(decl.bytes / 4);
        err = vm.getOutput(slot, out.data(), decl.bytes);
        if (!err.ok) { report(err); return 1; }

        std::cout << "output slot " << slot << " (" << decl.name << "):";
        for (size_t i = 0; i < out.size(); ++i)
            std::cout << " " << out[i];
        std::cout << std::endl;
    }

    return 0;
}
