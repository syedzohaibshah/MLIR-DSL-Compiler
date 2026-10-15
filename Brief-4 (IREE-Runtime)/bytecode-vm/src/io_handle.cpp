#include "io_handle.h"
#include "error.h"
#include <cstring>
#include <iostream>
#include <string>


//links the slot id reurned by the allocated buffer with the actual data
void IOHandler::setInput(int slot, void* data, int size){
    if(slot >= (int)input_slots.size()){
        input_slots.resize(slot + 1, {nullptr, 0});
    }
    input_slots[slot] = {data, size};
}



bool IOHandler::getOutput(int slot, void* out_ptr, int size){
    if(slot < 0 || slot >= (int)output_slots.size()){
        std::cerr<<"getOutput: output slot "<<slot<<" was never written by COPY_D2H"<<std::endl;
        return false;
    }
    if((int)output_slots[slot].size() != size){  //what is happening heree?
        std::cerr<<"getOutput: output slot "<<slot<<" holds "<<output_slots[slot].size()
                  <<" bytes, caller asked for "<<size<<std::endl;
        return false;
    }
    
    memcpy(out_ptr, output_slots[slot].data(), size); 
    return true;
 }


//  Is this slot registered? (not nullptr, not out of bounds)
// Does the registered size match what the opcode expects?
//"isnt reduntant?"
bool IOHandler::checkInputSlot(int slot, int expected_size, int line_number){
    if(slot < 0 || slot >= (int)input_slots.size() || input_slots[slot].data == nullptr){
        fail("input slot "+std::to_string(slot)+" was never set via vm_set_input", line_number);
        return false;
    }
    if(input_slots[slot].size != expected_size){
        fail("input slot "+std::to_string(slot)+" holds "+std::to_string(input_slots[slot].size)+
             " bytes, COPY_H2D asked for "+std::to_string(expected_size), line_number);
        return false;
    }
    return true;
}


void* IOHandler::inputData(int slot){  //Returns the HOST memory pointer for the input slot
    return input_slots[slot].data;
}



// Grows the output_slots vector if needed
// Resizes the specific slot's buffer to hold size bytes
// Returns a pointer to that buffer

uint8_t* IOHandler::prepareOutputSlot(int slot, int size){
    if(slot >= (int)output_slots.size()){
        output_slots.resize(slot + 1);
    }
    output_slots[slot].resize(size);
    return output_slots[slot].data();  //return pointer to the output slot where we have to store result
}
