#pragma once

#include "device.h"
#include "emunpu/device.h"   // external library header needed to specify path at compile time
#include <vector>
#include <string>

struct TensorSlot {
    emunpu_tensor_id_t real_id;
    bool freed;
};

class emunpuDevice:public Device{

    emunpu_device_t* emu_device;
    std::vector<TensorSlot> buffer_table;
    bool isValidHandle(DeviceHandle h, int line);

    //kernels

    bool do_matmul(int num_buffers,int num_scalars,
        int scalars_start,int buffer_start,const std::vector<int>& args,int line);

     bool do_vec_add(int num_buffers,int num_scalars,
         int scalars_start,int buffer_start,const std::vector<int>& args,int line);

     bool do_relu(int num_buffers,int num_scalars,
         int scalars_start,int buffer_start,const std::vector<int>& args,int line);


     bool  do_bais_add(int num_buffers,int num_scalars,
         int scalars_start,int buffer_start,const std::vector<int>& args,int line);
     
public:
    emunpuDevice();
    ~emunpuDevice() override;

    std::string name() const override;
     DeviceHandle allocate(int size,int line) override;
     void free(DeviceHandle handle,int line)override;
     bool upload(DeviceHandle handle, const void* src, int size, int line)override;
    bool download(DeviceHandle handle, void* dst, int size, int line) override;
    bool dispatch(const std::vector<int>& args, int line) override;
};