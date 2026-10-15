
// Same 2-layer network as main.cpp, targeting EmunpuDevice instead of
// CudaDevice. Only the LAUNCH lines differ between bytecode12.txt and
// bytecode12_emunpu.txt -- everything else in this file is identical to
// main.cpp on purpose, since the computation itself hasn't changed, only
// which hardware runs it.
//
// Expected final result: [0, 0, 8, 0]
#include <iostream>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"

int main(){

    Bytecode bytecode;

    if(!bytecode.loadFromFile("bytecode12_emunpu.txt")) return 1;

    std::cout<<"No of opcodes: "<<bytecode.size()<<std::endl;

    emunpuDevice npu;
    VM vm(&npu);

    float X[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float W0[4] = {2.0f, 0.0f, 0.0f, 2.0f};
    float B0[2] = {1.0f, 1.0f};
    float W1[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    float B1[2] = {-8.0f, -8.0f};

    bytecode.checkInputSize(0, sizeof(X));
    bytecode.checkInputSize(1, sizeof(W0));
    bytecode.checkInputSize(2, sizeof(B0));
    bytecode.checkInputSize(3, sizeof(W1));
    bytecode.checkInputSize(4, sizeof(B1));

    vm.setInput(0, X,  sizeof(X));
    vm.setInput(1, W0, sizeof(W0));
    vm.setInput(2, B0, sizeof(B0));
    vm.setInput(3, W1, sizeof(W1));
    vm.setInput(4, B1, sizeof(B1));

    vm.execute(bytecode.get_instructions());

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    bytecode.checkOutputSize(0, sizeof(result));

    vm.getOutput(0, result, sizeof(result));

    std::cout<<"Result (2-layer network, chained + looped, emunpu): ";
    for(float v : result) std::cout<<v<<" ";
    std::cout<<std::endl;
    std::cout<<"Expected:                                            0 0 8 0"<<std::endl;

    return 0;
}


// clang++ -std=c++14 -I src -I src/hal -I ../MLIRDSLProject/emunpu-sim/include \
//   src/bytecode.cpp src/io_handle.cpp src/vm_core.cpp src/hal/emunpu_device.cpp \
//   src/main_emunpu.cpp \
//   -L ../MLIRDSLProject/emunpu-sim/build -lemunpu -lm \
//   -o emunpu_test
