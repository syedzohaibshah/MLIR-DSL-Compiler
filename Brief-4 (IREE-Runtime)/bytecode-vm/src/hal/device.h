#pragma once


#include <cstdint>
#include <cstddef>
#include <vector>
#include<string>

using DeviceHandle = int;

class Device{
    
public:

    virtual ~Device()=default;
    virtual std::string name() const = 0;
    virtual DeviceHandle allocate(int size,int line)=0;
    virtual void free(DeviceHandle handle,int line)=0;
    virtual bool upload(DeviceHandle handle, const void* src, int size, int line)=0;
    virtual bool download(DeviceHandle handle, void* dst, int size, int line) =0;
    virtual bool dispatch(const std::vector<int>& args, int line) =0;


};