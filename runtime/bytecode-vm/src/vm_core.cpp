#include "vm_core.h"
#include "error.h"
#include <iostream>




VM::VM(Device* device) : device(device) {}


Error VM::setInput(int slot, void* data, int size){
    return io.setInput(slot, data, size);
}

Error VM::getOutput(int slot, void* out_ptr, int size){
    return io.getOutput(slot, out_ptr, size);
}

Error VM::execute(const std::vector<opcode>& opcodes){
     int next_index;
    int opcode_size=opcodes.size();

    // Step limit: stops a runaway loop (for example, a wrong COND_BR that
    // jumps back forever). A real run re-runs instructions many times, so the
    // ceiling is large. 
    long long executed = 0;
    const long long step_limit = 1e8; // 1e8

    for(int i=0;i<opcode_size;){
         if(++executed > step_limit){
             return makeError("step limit exceeded - possible infinite loop", opcodes[i].line_number);
         }
         auto& op=opcodes[i];
          next_index=-1; //reset next_index for each opcode
          RETURN_IF_ERROR(dispatch(op.name, op.args, op.line_number,i,opcode_size,next_index));
         if(next_index>opcode_size || next_index<0) {

             return makeError("next_index out of bounds", op.line_number);
         }
         i=next_index; //jump to target index

    }
    return ok();
}

// COPY_H2D buffer_id input_slot size
Error VM::copyHostToDevice(int desti, int src_slot, int size, int line_number){

 RETURN_IF_ERROR(io.checkInputSlot(src_slot, size, line_number));//check if this input slot is valid and  size is alos consistent



   RETURN_IF_ERROR(device->upload(desti, io.inputData(src_slot), size, line_number));

    return ok();
}

// COPY_D2H output_slot buffer_id size
Error VM::copyDeviceToHost(int dest_slot, int src, int size, int line_number) {

    uint8_t* out_buf = io.prepareOutputSlot(dest_slot, size); //resize output slot and return the pointer to it


     if(out_buf == nullptr) {
        return makeError(" output slot cannot be negative", line_number);
    }

    RETURN_IF_ERROR(device->download(src, out_buf, size, line_number)); 
    io.markOutputSlotWritten(dest_slot);
    return ok();
}

Error VM::dispatch(const std::string& name, const std::vector<Arg>& args, int line_number,int index,int opcode_size, int& next_index){
    

    //value
    if(name=="ALLOC" && args.size()==1){

           int alloc_size;
           RETURN_IF_ERROR(resolveValue(args[0], alloc_size, line_number));

        if(alloc_size <= 0){
            return makeError("ALLOC has zero or  negative size", line_number);
        }
        DeviceHandle handle;
        RETURN_IF_ERROR(device->allocate(alloc_size, line_number,handle));
        
         next_index = index+1;
         return ok();
    }
    else if(name=="FREE" && args.size()==1){ //value 
            int buff_id;
           RETURN_IF_ERROR(resolveValue(args[0], buff_id, line_number));
        
        RETURN_IF_ERROR(device->free(buff_id, line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="COPY_H2D" && args.size()==3){  //value
     
      
        int desti_buf_slot;
        RETURN_IF_ERROR(resolveValue(args[0], desti_buf_slot, line_number));

        
        int src_input_slot;
            RETURN_IF_ERROR(resolveValue(args[1], src_input_slot, line_number));


        int copy_size;
        RETURN_IF_ERROR(resolveValue(args[2], copy_size, line_number));

        RETURN_IF_ERROR(copyHostToDevice(desti_buf_slot, src_input_slot, copy_size, line_number));

        next_index = index+1;
        return ok();
    }
    else if(name=="COPY_D2H" && args.size()==3){ //value

        int desti_out_slot;
        RETURN_IF_ERROR(resolveValue(args[0], desti_out_slot, line_number));
      

        int src_buf_slot;
        RETURN_IF_ERROR(resolveValue(args[1], src_buf_slot, line_number));
       
        
        int copy_size;
        RETURN_IF_ERROR(resolveValue(args[2], copy_size, line_number));

        RETURN_IF_ERROR(copyDeviceToHost(desti_out_slot, src_buf_slot, copy_size, line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="LAUNCH"){//value

        std::vector<int>int_args;
        for(auto arg: args){
            int arg_value=-1;
            RETURN_IF_ERROR(resolveValue(arg, arg_value, line_number));
         
            int_args.push_back(arg_value);
        }
       // kernels.launch(int_args, line_number, buffers);
        RETURN_IF_ERROR(device->dispatch(int_args, line_number));
         next_index = index+1;
            return ok();
    }
    else if(name=="SET" && args.size()==2){ //reg_id is id while value can be reg
  
       int set_value;
       RETURN_IF_ERROR(resolveValue(args[1], set_value, line_number));
        
       if(args[0].is_register==true){
           return makeError("SET first argument is already a register id", line_number);
       }
       RETURN_IF_ERROR(writeRegister(args[0].value,set_value,line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="DEC" && args.size()==1){ //id

       if(args[0].is_register==true){
           return makeError("DEC  argument is already a register id", line_number);
       }
      
        RETURN_IF_ERROR(decrementRegister(args[0].value,line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="INC" && args.size()==1){//id
        if(args[0].is_register==true){
           return makeError("INC  argument is already a register id", line_number);
       }
    
        RETURN_IF_ERROR(incrementRegister(args[0].value,line_number));
        next_index = index+1;
        return ok();
    }
    else if (name=="BR" && args.size()==1){
        //just jump to the target_index

        if(args[0].is_register==true){
           return makeError("BR argument is a instruction line cannot be a register", line_number);
       }
        next_index = args[0].value;
        //out of bounds check
        if(next_index<0 || next_index>=opcode_size){
            return makeError("BR target index out of bounds", line_number);
        }
        return ok();
    }
    else if(name=="COND_BR" && args.size()==3){//ids

        //ist arguemnt is already a register id, second and third argument are instruction line numbers
        if(args[0].is_register==true || args[1].is_register==true || args[2].is_register==true){
            return makeError("COND_BR first argument is already a register id, second and third arguments cannot be registers as they are instruction lines", line_number);
        }

        int reg_value;
        RETURN_IF_ERROR(readRegister(args[0].value,reg_value,line_number));

        next_index = reg_value==0?  args[1].value: args [2].value;

        //out of bounds check
        if(next_index<0 || next_index>=opcode_size){
            return makeError("COND_BR target index out of bounds", line_number);
        }

        return ok();
        
    }
    else if(name=="EOF" && args.size()==0){
        std::cout<<"End of file"<<std::endl;
        next_index = opcode_size; //jump to the end of the opcode list
        return ok();
    }
    else{

        next_index = index+1;
        return makeError("unknown opcode or incorrect number of arguments " + name, line_number);
    }
}


Error VM::readRegister(int reg_id,int &value,int line_number){

    if (reg_id >= MAX_REGISTERS ||   reg_id < 0 ) {
 
         return makeError("out of range register id", line_number);
    }
    
    auto it = registers.find(reg_id);
    if (it == registers.end()) {
        return makeError("register id not found", line_number);
    }

    value= it->second;
    return ok();
}

Error VM::writeRegister(int reg_id,int value,int line_number){



    if(reg_id<0 || reg_id >= MAX_REGISTERS){ 
        return makeError("out of range register id", line_number);
    }
    
    registers[reg_id]=value;
    return ok();
}


Error VM:: decrementRegister(int reg_id,int line_number){

int value;
RETURN_IF_ERROR(readRegister(reg_id,value,line_number));

if(value==INT_MIN){
    return makeError("register value underflow", line_number);
}
//value can go negative
RETURN_IF_ERROR(writeRegister(reg_id,value-1, line_number));

return ok();
 }

 Error VM:: incrementRegister(int reg_id,int line_number){

     
     int value;
    RETURN_IF_ERROR(readRegister(reg_id,value,line_number));
    
    if(value==INT_MAX){
        return makeError("register value overflow", line_number);
    }
    
    RETURN_IF_ERROR(writeRegister(reg_id,value+1, line_number));
    return ok();

 }



 // if arg.is_register, read the register; else use the literal.
Error VM:: resolveValue(const Arg& arg, int& out, int line){
    
    if(arg.is_register){
        RETURN_IF_ERROR(readRegister(arg.value,out,line));
    }else{
        out=arg.value;
    }

    return ok();
}
