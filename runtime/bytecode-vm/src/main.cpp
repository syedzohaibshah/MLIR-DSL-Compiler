
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
#include "hal/cuda_device.h"

int main(){

    Bytecode bytecode;

    if(!bytecode.loadFromFile("bytecode12.txt")) return 1;

    std::cout<<"No of opcodes: "<<bytecode.size()<<std::endl;

    cudaDevice gpu;
    VM vm(&gpu);

   

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

    std::cout<<"Result (2-layer network, chained + looped): ";
    for(float v : result) std::cout<<v<<" ";
    std::cout<<std::endl;
    std::cout<<"Expected:                                    0 0 8 0"<<std::endl;

    return 0;
}
