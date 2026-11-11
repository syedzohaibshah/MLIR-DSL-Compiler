
// Layer 0: x=[[1,2],[3,4]], W0=[[2,0],[0,2]], B0=[1,1]
//   matmul  = x @ W0        = [[2,4],[6,8]]
//   biased  = matmul + B0   = [[3,5],[7,9]]
//   relu    = ReLU(biased)  = [[3,5],[7,9]]   (nothing negative)
//
// Layer 1: input = layer 0's output, W1=[[1,-1],[1,-1]], B1=[-8,-8]
//   matmul  = input @ W1    = [[8,-8],[16,-16]]
//   biased  = matmul + B1   = [[0,-16],[8,-24]]
//   relu    = ReLU(biased)  = [[0,0],[8,0]]
//
// Expected final result: [0, 0, 8, 0]
#include <iostream>
#include "bytecode.h"
#include "vm_core.h"
#include "error.h"
#include "hal/cuda_device.h"

// Same shape as main_emunpu.cpp: every call returns an Error, and the first
// Error stops the run. main() prints it.
Error run(){

    Bytecode bytecode;

    RETURN_IF_ERROR(bytecode.loadFromFile("bytecode12.txt"));

    std::cout<<"No of opcodes: "<<bytecode.size()<<std::endl;

    cudaDevice gpu;
    VM vm(&gpu);
    RETURN_IF_ERROR(vm.bindSignature(bytecode));

    float X[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float W0[4] = {2.0f, 0.0f, 0.0f, 2.0f};
    float B0[2] = {1.0f, 1.0f};
    float W1[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    float B1[2] = {-8.0f, -8.0f};

    RETURN_IF_ERROR(vm.setInput(0, X,  sizeof(X)));
    RETURN_IF_ERROR(vm.setInput(1, W0, sizeof(W0)));
    RETURN_IF_ERROR(vm.setInput(2, B0, sizeof(B0)));
    RETURN_IF_ERROR(vm.setInput(3, W1, sizeof(W1)));
    RETURN_IF_ERROR(vm.setInput(4, B1, sizeof(B1)));

    RETURN_IF_ERROR(vm.execute(bytecode.get_instructions()));

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    RETURN_IF_ERROR(vm.getOutput(0, result, sizeof(result)));

    std::cout<<"Result (2-layer network, chained + looped): ";
    for(float v : result) std::cout<<v<<" ";
    std::cout<<std::endl;
    std::cout<<"Expected:                                    0 0 8 0"<<std::endl;

    return ok();
}

int main(){
    Error err = run();
    if(!err.ok){
        report(err);
        return 1;
    }
    return 0;
}
