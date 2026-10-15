#include "cuda_device.h"

#include "../error.h"
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"

cudaDevice::cudaDevice() { registerKernels(); }


DeviceHandle cudaDevice::allocate(int size,int line_number){

    void* dev_ptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fail("cudaMalloc failed with error: "+std::string(cudaGetErrorString(err)), line_number);
        return -1;
    }
    int id = (int)buffer_table.size();
    buffer_table.push_back({size, false, dev_ptr}); //keeing track of the device allocate data
    std::cout<<"[cuda] alloc buffer "<<id<<" ("<<size<<" bytes)"<<std::endl;

    return id;
}

bool cudaDevice::isValidHandle(DeviceHandle h, int line) {
    if (h < 0 || h >= (int)buffer_table.size()) {
        fail("invalid buffer id", line);
        return false;
    }
    if (buffer_table[h].freed) {
        fail("buffer already freed", line);
        return false;
    }
    return true;
}



void cudaDevice:: free(DeviceHandle  buffer_id,int line_number){

    if(isValidHandle(buffer_id, line_number)){
        cudaError_t err = cudaFree(buffer_table[buffer_id].dev_ptr);
        if(err != cudaSuccess){
            fail("cudaFree failed with error: "+std::string(cudaGetErrorString(err)), line_number);
            return;
        }
        buffer_table[buffer_id].freed = true;
        buffer_table[buffer_id].dev_ptr = nullptr;
        std::cout<<"[cuda] free buffer "<<buffer_id<<std::endl;
          }

        
    }


 bool cudaDevice:: upload(DeviceHandle buffer_id, const void* src, int size, int line_number){

             if(!isValidHandle(buffer_id, line_number)) return false; //check if the buffer id is valid 
       
             cudaError_t err = cudaMemcpy(devicePtr(buffer_id), src, size, cudaMemcpyHostToDevice);
             if(err != cudaSuccess){
                 fail("cudaMemcpy (H2D) failed with error: "+std::string(cudaGetErrorString(err)), line_number);
                 return false ;
             }
             std::cout<<"[cuda] upload "<<size<<" bytes -> buffer "<<buffer_id<<std::endl;
             return true;
         }


bool cudaDevice:: download(DeviceHandle buffer_id, void* dst, int size, int line_number){

                 if(!isValidHandle(buffer_id, line_number)) return false;
             
 
                 cudaError_t err = cudaMemcpy(dst, devicePtr(buffer_id), size, cudaMemcpyDeviceToHost);
                 if(err != cudaSuccess){
                     fail("cudaMemcpy (D2H) failed with error: "+std::string(cudaGetErrorString(err)), line_number);
                     return false;
                 }
                 std::cout<<"[cuda] download "<<size<<" bytes <- buffer "<<buffer_id<<std::endl;
                 return true;
             }


             
    std::string cudaDevice::name() const {
        return "cuda";
    }



    void cudaDevice::registerKernels(){
        // kernel 0: vec_mm_Kernel - 3 buffers (x, y, z), 3 scalars (n, width, height)
        kernel_table[0] = [this](const KernelContext& ctx){
            return launchGenericKernel("vec_mm_Kernel", (void*)vec_mm_Kernel, 3, 3, ctx);
        };
        // kernel 1: vec_add_Kernel - 3 buffers (x, y, z), 1 scalar (n)
        kernel_table[1] = [this](const KernelContext& ctx){
            return launchGenericKernel("vec_add_Kernel", (void*)vec_add_Kernel, 3, 1, ctx);
        };
        // kernel 2: relu_Kernel - 2 buffers (x, z), 1 scalar (n)
        kernel_table[2] = [this](const KernelContext& ctx){
            return launchGenericKernel("relu_Kernel", (void*)relu_Kernel, 2, 1, ctx);
        };
        // kernel 3: bias_add_Kernel - 3 buffers (x, bias, z), 2 scalars (n, out_features)
        kernel_table[3] = [this](const KernelContext& ctx){
            return launchGenericKernel("bias_add_Kernel", (void*)bias_add_Kernel, 3, 2, ctx);
        };
    }

bool  cudaDevice:: launchGenericKernel(const char* name,
                   void* kernel_fn,
                   int expected_buffers,
                   int expected_scalars,
                   const KernelContext& ctx){

    if(ctx.num_buffers != expected_buffers){
        fail(std::string(name)+" expects exactly "+std::to_string(expected_buffers)+" buffers", ctx.line_number);
        return false;
    }
    if(ctx.num_scalars != expected_scalars){
        fail(std::string(name)+" expects exactly "+std::to_string(expected_scalars)+" scalars", ctx.line_number);
        return false;
    }

    const auto& args = *ctx.args;

    std::vector<float*> buffer_ptrs(expected_buffers);
    for(int i = 0; i < expected_buffers; i++){
        buffer_ptrs[i] = (float*)devicePtr(args[ctx.buffers_start + i]);
    }

    std::vector<int> scalars(expected_scalars);
    for(int i = 0; i < expected_scalars; i++){
        scalars[i] = args[ctx.scalars_start + i];
    }

    // cudaLaunchKernel wants an array of pointers-TO-each-argument,
    std::vector<void*> kernel_args;
    for(auto& p : buffer_ptrs) kernel_args.push_back(&p);
    for(auto& s : scalars) kernel_args.push_back(&s);

    cudaError_t err = cudaLaunchKernel(kernel_fn, ctx.grid, ctx.block, kernel_args.data(), 0, 0);
    if(err != cudaSuccess){
        fail(std::string(name)+" launch failed: "+std::string(cudaGetErrorString(err)), ctx.line_number);
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();  // wait for the kernel to execute
    if(sync_err != cudaSuccess){
        fail(std::string(name)+" execution failed: "+std::string(cudaGetErrorString(sync_err)), ctx.line_number);
        return false;
    }

    std::cout<<name<<" completed"<<std::endl;
    return true;
    }

      bool cudaDevice::dispatch(const std::vector<int>& args, int line_number) {

          if(args.size() < 2) {
              fail("LAUNCH expected at least kernel_id and num_buffers", line_number); return false;
          }
      
          int kernel_id = args[0];
          int num_buffers = args[1];
      
          if(num_buffers < 0) { fail("LAUNCH has negative num_buffers", line_number); return false; }
      
          size_t dims_start = 2 + (size_t)num_buffers;
          size_t num_scalars_index = dims_start + 6;
          if(args.size() <= num_scalars_index) { fail("LAUNCH missing dimensions or num_scalars", line_number); return false; }
      
          int num_scalars = args[num_scalars_index];
          if(num_scalars < 0) { fail("LAUNCH has negative num_scalars", line_number); return false; }
      
          size_t expected_total = num_scalars_index + 1 + (size_t)num_scalars;
          if(args.size() != expected_total) { fail("LAUNCH arg count mismatch", line_number); return false; }
      
          for(size_t i = 2; i < dims_start; ++i){
              if(!isValidHandle(args[i], line_number)) return false;
          }
      
          std::cout<<"Launching kernel ID "<<kernel_id<<" with "<<num_buffers<<" buffers, grid("
                    <<args[dims_start]<<","<<args[dims_start+1]<<","<<args[dims_start+2]<<"), block("
                    <<args[dims_start+3]<<","<<args[dims_start+4]<<","<<args[dims_start+5]<<"), "
                    <<num_scalars<<" scalar args"<<std::endl;
      
          auto it = kernel_table.find(kernel_id);
          if(it == kernel_table.end()){
              fail("unknown kernel_id ", line_number);
              return false;
          }
      
          KernelContext ctx; //prepare the context for kernel to execute
          ctx.args = &args;
          ctx.buffers_start = 2;
          ctx.num_buffers = num_buffers;
          ctx.grid = dim3(args[dims_start], args[dims_start+1], args[dims_start+2]);
          ctx.block = dim3(args[dims_start+3], args[dims_start+4], args[dims_start+5]);
          ctx.scalars_start = num_scalars_index + 1;
          ctx.num_scalars = num_scalars;
       //   ctx.buffers = &buffers;
          ctx.line_number = line_number;
      
        return  it->second(ctx);  // looks up the registered lambda and calls it with ctx


      }


      
      
      //it will be called in the COPY with the buffer id and it will return the device pointer we set in  alloc ,
      void* cudaDevice::devicePtr(int id){
          return buffer_table[id].dev_ptr;
      }