#pragma once
#include <vector>
#include <string>
#include "bytecode.h"
#include "io_handle.h"
#include "hal/device.h"

class VM{
public:
    void setInput(int slot, void* data, int size);
    bool getOutput(int slot, void* out_ptr, int size);
    void execute(const std::vector<opcode>& opcodes);
    VM(Device* device);

private:
  
    IOHandler io;
    Device *device;
    std::vector<int>registers;

    void copyHostToDevice(int desti, int src_slot, int size, int line_number);
    void copyDeviceToHost(int dest_slot, int src, int size, int line_number);
    int dispatch(const std::string& name, const std::vector<Arg>& args, int line_number,int index,int opcode_size);


    //registers
    void writeRegister(int id,int value,int line_number);
    bool readRegister(int id,int &value,int line_number);
     void decrementRegister(int reg_id,int line_number);
      void incrementRegister(int reg_id,int line_number);
};
