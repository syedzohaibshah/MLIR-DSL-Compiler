
// Same 2-layer network as main.cpp, targeting EmunpuDevice instead of
// CudaDevice. Only the LAUNCH lines differ between bytecode12.txt and
// bytecode12_emunpu.txt 
// Expected final result: [0, 0, 8, 0]
#include <iostream>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"
#include "error.h"
#include "io_handle.h"
#include "runtime_limits.h"

Error run( std::string filename ){


    Bytecode bytecode;

    RETURN_IF_ERROR(bytecode.loadFromFile(filename));
 

    emunpuDevice npu;
    RETURN_IF_ERROR(npu.status());  // Check if the device was created successfully
    RuntimeLimits runtime_limits;
    VM vm(&npu, runtime_limits);
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

    std::cout<<"Result (2-layer network, chained + looped, emunpu): ";
    for(float v : result) std::cout<<v<<" ";
    std::cout<<std::endl;
    std::cout<<"Expected:                                            0 0 8 0"<<std::endl;

    return ok();

}




int main(int argc, char** argv){

    std::string filename;
    if(argc>1){
 filename = argv[1];
    }
    else {
        std::cerr<<"provide bytecode file"<<std::endl;
        return 1;
    }

Error err=run(filename);
if(!err.ok){
    report(err);
    return 1;
}


return 0;
}

