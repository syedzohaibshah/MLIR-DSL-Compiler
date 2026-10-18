
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
#include "error.h"

Error run(){


    Bytecode bytecode;

    RETURN_IF_ERROR(bytecode.loadFromFile("bytecode12_emunpu.txt"));
 

    emunpuDevice npu;
    RETURN_IF_ERROR(npu.status());  // Check if the device was created successfully
    VM vm(&npu);

    float X[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float W0[4] = {2.0f, 0.0f, 0.0f, 2.0f};
    float B0[2] = {1.0f, 1.0f};
    float W1[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    float B1[2] = {-8.0f, -8.0f};

    RETURN_IF_ERROR(bytecode.checkInputSize(0, sizeof(X)));
    RETURN_IF_ERROR(bytecode.checkInputSize(1, sizeof(W0)));
    RETURN_IF_ERROR(bytecode.checkInputSize(2, sizeof(B0)));
    RETURN_IF_ERROR(bytecode.checkInputSize(3, sizeof(W1)));
    RETURN_IF_ERROR(bytecode.checkInputSize(4, sizeof(B1)));

    RETURN_IF_ERROR(vm.setInput(0, X,  sizeof(X)));
    RETURN_IF_ERROR(vm.setInput(1, W0, sizeof(W0)));
    RETURN_IF_ERROR(vm.setInput(2, B0, sizeof(B0)));
    RETURN_IF_ERROR(vm.setInput(3, W1, sizeof(W1)));
    RETURN_IF_ERROR(vm.setInput(4, B1, sizeof(B1)));

    RETURN_IF_ERROR(vm.execute(bytecode.get_instructions()));

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    RETURN_IF_ERROR(bytecode.checkOutputSize(0, sizeof(result)));

    RETURN_IF_ERROR(vm.getOutput(0, result, sizeof(result)));

    std::cout<<"Result (2-layer network, chained + looped, emunpu): ";
    for(float v : result) std::cout<<v<<" ";
    std::cout<<std::endl;
    std::cout<<"Expected:                                            0 0 8 0"<<std::endl;

    return ok();

}




int main(){

Error err=run();
if(!err.ok){
    report(err);
    return 1;
}


return 0;
}

