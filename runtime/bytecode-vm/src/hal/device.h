#pragma once


#include <cstdint>
#include <cstddef>
#include <vector>
#include<string>
#include "../error.h"

using DeviceHandle = int;

struct DeviceLimits {
    int64_t max_allocation_bytes;   // largest single ALLOC
    int64_t total_memory_bytes;     // whole device capacity
    int     alignment_bytes;        // emunpu: 4 (fp32)
    int     max_live_buffers;       // 0 = device sets no limit
};


class Device{
    
public:

    virtual ~Device()=default;
  
    virtual std::string name() const = 0;
    virtual DeviceLimits limits() const = 0;
    
    // Handle contract: give out handles 0, 1, 2, ... in call order, and never
    // reuse the handle of a freed buffer. Reusing the MEMORY is allowed; only
    // the handle stays unique. The VM indexes its buffer table by handle and
    // verifies this on every ALLOC. See doc/validation-design.md section 7.
    virtual Error allocate(int size,int line,DeviceHandle& out_handle)=0;
    virtual Error free(DeviceHandle handle,int line)=0;
    virtual Error upload(DeviceHandle handle, const void* src, int size, int line)=0;
    virtual Error download(DeviceHandle handle, void* dst, int size, int line) =0;
    virtual Error dispatch(const std::vector<int>& args, int line) =0;
    

};