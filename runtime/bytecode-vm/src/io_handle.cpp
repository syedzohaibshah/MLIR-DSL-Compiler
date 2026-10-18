#include "io_handle.h"
#include "error.h"
#include <cstring>
#include <iostream>
#include <string>


//links the slot id reurned by the allocated buffer with the actual data
Error IOHandler::setInput(int slot, void* data, int size){
   
    if(slot >= (int)input_slots.size()){
        input_slots.resize(slot + 1, {nullptr, 0});
    }
    input_slots[slot] = {data, size};
    return ok();
}



Error IOHandler::getOutput(int slot, void* out_ptr, int size){
    if(slot < 0 || slot >= (int)output_slots.size() || !output_slots[slot].written_ok){
       // std::cerr<<"getOutput: output slot "<<slot<<" is either out of range or  never written cleanly by D2H"<<std::endl;
        return makeError("output slot not ready", 0);
    }
    if((int)output_slots[slot].data.size() != size){  //what is happening heree?
       // std::cerr<<"getOutput: output slot "<<slot<<" holds "<<output_slots[slot].size()
                 // <<" bytes, caller asked for "<<size<<std::endl;
        return makeError("output slot size mismatch",  0);
    }
    
    memcpy(out_ptr, output_slots[slot].data.data(), size); 
    return ok();
 }


//  Is this slot registered? (not nullptr, not out of bounds)
// Does the registered size match what the opcode expects?
//"isnt reduntant?"
Error IOHandler::checkInputSlot(int slot, int expected_size, int line_number){
    if(slot < 0 || slot >= (int)input_slots.size() || input_slots[slot].data == nullptr){
        return makeError("input slot "+std::to_string(slot)+" was never set via vm_set_input",line_number);
    }
    if(input_slots[slot].size != expected_size){
        return makeError("input slot "+std::to_string(slot)+" holds "+std::to_string(input_slots[slot].size)+
             " bytes, COPY_H2D asked for "+std::to_string(expected_size),line_number);
    }
    return ok();
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
    output_slots[slot].data.resize(size);
    output_slots[slot].written_ok = false;  //will be set to true after a clean download
    return output_slots[slot].data.data();  //return pointer to the output slot where we have to store result
}

 void IOHandler::markOutputSlotWritten(int slot){
        if (slot >= 0 && slot < (int)output_slots.size()) {
            output_slots[slot].written_ok = true;
        }
    }