#pragma once


#include <cstdint>
#include <cstddef>
#include <vector>
#include<string>
#include "../error.h"

using DeviceHandle = int;

class Device{
    
public:

    virtual ~Device()=default;
  
    virtual std::string name() const = 0;
    
    virtual Error allocate(int size,int line,DeviceHandle& out_handle)=0;
    virtual Error free(DeviceHandle handle,int line)=0;
    virtual Error upload(DeviceHandle handle, const void* src, int size, int line)=0;
    virtual Error download(DeviceHandle handle, void* dst, int size, int line) =0;
    virtual Error dispatch(const std::vector<int>& args, int line) =0;


};