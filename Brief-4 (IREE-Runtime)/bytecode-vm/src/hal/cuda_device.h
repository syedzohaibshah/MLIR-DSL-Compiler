#pragma once 
#include "device.h"
#include <vector>
#include <functional>
#include <unordered_map>
#include <cuda_runtime.h>

struct Buffer{
    int size;
    bool freed;
    void* dev_ptr;  //Pointer to the allocated device memory
};


// prepare Everything kernel needs
struct KernelContext{
    const std::vector<int>* args;   // raw LAUNCH args, in case  needed
    size_t buffers_start;           // args[buffers_start .. buffers_start+num_buffers-1] = buffer IDs
    int num_buffers;
    dim3 grid;
    dim3 block;
    size_t scalars_start;           // args[scalars_start .. scalars_start+num_scalars-1] = scalars
    int num_scalars;
  // BufferManager* buffers;    //extracted data from buffers base in ids
    int line_number;
};


using KernelHandler = std::function<bool(const KernelContext&)>;

class cudaDevice:public Device{
  std::vector<Buffer> buffer_table;
  //it will map kernel ids to the kernel functions
  std::unordered_map<int, KernelHandler> kernel_table;

  
  
      bool isValidHandle(DeviceHandle h, int line);
          void* devicePtr(int id);
          void registerKernels();
          bool launchGenericKernel(const char* name,
                             void* kernel_fn,
                             int expected_buffers,
                             int expected_scalars,
                             const KernelContext& ctx);
    

public: 

    cudaDevice();
    std::string name() const override;
     DeviceHandle allocate(int size,int line) override;
     void free(DeviceHandle handle,int line)override;
     bool upload(DeviceHandle handle, const void* src, int size, int line)override;
    bool download(DeviceHandle handle, void* dst, int size, int line) override;
    bool dispatch(const std::vector<int>& args, int line) override;

 
    
};