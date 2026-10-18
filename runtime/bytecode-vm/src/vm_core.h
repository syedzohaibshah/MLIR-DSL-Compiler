#pragma once
#include <vector>
#include <string>
#include "bytecode.h"
#include "io_handle.h"
#include "hal/device.h"

class VM{
public:
    Error setInput(int slot, void* data, int size);
    Error getOutput(int slot, void* out_ptr, int size);
    Error execute(const std::vector<opcode>& opcodes);
    VM(Device* device);

private:
  
    IOHandler io;
    Device *device;
    std::vector<int>registers;

    Error copyHostToDevice(int desti, int src_slot, int size, int line_number);
    Error copyDeviceToHost(int dest_slot, int src, int size, int line_number);
    Error dispatch(const std::string& name, const std::vector<Arg>& args,
               int line_number, int index, int opcode_size, int& next_index);


    //registers
    Error writeRegister(int id,int value,int line_number);
    Error readRegister(int id,int &value,int line_number);
     Error decrementRegister(int reg_id,int line_number);
      Error incrementRegister(int reg_id,int line_number);
};
