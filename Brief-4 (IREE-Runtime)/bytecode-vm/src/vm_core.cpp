#include "vm_core.h"
#include "error.h"
#include <iostream>




VM::VM(Device* device) : device(device) {}


void VM::setInput(int slot, void* data, int size){
    io.setInput(slot, data, size);
}

bool VM::getOutput(int slot, void* out_ptr, int size){
    return io.getOutput(slot, out_ptr, size);
}

void VM::execute(const std::vector<opcode>& opcodes){

    int opcode_size=opcodes.size();
    for(int i=0;i<opcode_size;){
         auto& op=opcodes[i];
         int target_index = dispatch(op.name, op.args, op.line_number,i,opcode_size);
         if(target_index>opcode_size) {
             fail("out of range index", op.line_number);
             return;
         }
         i=target_index; //jump to target index
         
    }
}

// COPY_H2D buffer_id input_slot size
void VM::copyHostToDevice(int desti, int src_slot, int size, int line_number){
  // if(!buffers.isValid(desti, line_number)) return; //check if the buffer id is valid 
    
    if(!io.checkInputSlot(src_slot, size, line_number)) return;  //check if this input slot is valid and  size is alos consistent


    device->upload(desti, io.inputData(src_slot), size, line_number);

    //cudaError_t err = cudaMemcpy(buffers.devicePtr(desti), io.inputData(src_slot), size, cudaMemcpyHostToDevice);
    // if(err != cudaSuccess){
    //     fail("cudaMemcpy (H2D) failed with error: "+std::string(cudaGetErrorString(err)), line_number);
    //     return;
    // }
  //  std::cout<<"Copying "<<size<<" bytes: input slot "<<src_slot<<" -> buffer "<<desti<<std::endl;
}

// COPY_D2H output_slot buffer_id size
void VM::copyDeviceToHost(int dest_slot, int src, int size, int line_number) {
  //  if(!buffers.isValid(src, line_number)) return;

    uint8_t* out_buf = io.prepareOutputSlot(dest_slot, size); //resize output slot and return the pointer to it

    device->download(src, out_buf, size, line_number); 
    // cudaError_t err = cudaMemcpy(out_buf, buffers.devicePtr(src), size, cudaMemcpyDeviceToHost);
    // if(err != cudaSuccess){
    //     fail("cudaMemcpy (D2H) failed with error: "+std::string(cudaGetErrorString(err)), line_number);
    //     return;
    // }
    // std::cout<<"Copying "<<size<<" bytes: buffer "<<src<<" -> output slot "<<dest_slot<<std::endl;
}

int  VM::dispatch(const std::string& name, const std::vector<Arg>& args, int line_number,int index,int opcode_size){
    
    if(name=="ALLOC" && args.size()==1){
        //buffers.alloc(args[0].value, line_number);
        device->allocate(args[0].value, line_number);
        return index+1;
    }
    else if(name=="FREE" && args.size()==1){
        //buffers.free(args[0].value, line_number);
        device->free(args[0].value, line_number);
        return index+1;
    }
    else if(name=="COPY_H2D" && args.size()==3){
     
        // copyHostToDevice(args[0].value, args[1].value, args[2].value, line_number);
        
        int desti_buf_slot;
        if(args[0].is_register){
            if(!readRegister(args[0].value, desti_buf_slot, line_number)) return opcode_size+1;
        } else {
            desti_buf_slot = args[0].value;
        }
        
        int src_input_slot;
        if(args[1].is_register){
            if(!readRegister(args[1].value, src_input_slot, line_number)) return opcode_size+1;
        } else {
            src_input_slot = args[1].value;
        }
        
        copyHostToDevice(desti_buf_slot, src_input_slot, args[2].value, line_number);

        return index+1;
    }
    else if(name=="COPY_D2H" && args.size()==3){

        int desti_out_slot;
        if(args[0].is_register){
            if(!readRegister(args[0].value, desti_out_slot, line_number)) return opcode_size+1;
        }else{
            desti_out_slot=args[0].value;
        }

        int src_buf_slot;
        if(args[1].is_register){
            if(!readRegister(args[1].value, src_buf_slot, line_number)) return opcode_size+1;
        }else{
            src_buf_slot=args[1].value;
        }
        
        copyDeviceToHost(desti_out_slot, src_buf_slot, args[2].value, line_number);
        return index+1;
    }
    else if(name=="LAUNCH"){

        std::vector<int>int_args;
        for(auto arg: args){
            int arg_value=-1;
            if(arg.is_register){
                if(!readRegister(arg.value, arg_value, line_number)) return opcode_size+1;
            }else{
                arg_value=arg.value;
            }
            int_args.push_back(arg_value);
        }
       // kernels.launch(int_args, line_number, buffers);
        device->dispatch(int_args, line_number);
        return index+1;
    }
    else if(name=="SET" && args.size()==2){
        writeRegister(args[0].value,args[1].value,line_number);
        return index+1;
    }
    else if(name=="DEC" && args.size()==1){
        
        decrementRegister(args[0].value,line_number);
        return index+1;
    }
    else if(name=="INC" && args.size()==1){
        incrementRegister(args[0].value,line_number);
        return index+1;
        
    }
    else if (name=="BR" && args.size()==1){
        //just jump to the target_index
        return args[0].value;
    }
    else if(name=="COND_BR" && args.size()==3){
        int reg_value;
        if(!readRegister(args[0].value,reg_value,line_number)) return opcode_size+1 ; //is it right?
        return reg_value==0?  args[1].value: args [2].value;
        
    }
    else if(name=="EOF" && args.size()==0){
        std::cout<<"End of file"<<std::endl;
        return index+1;
    }
    else{
        fail("either Unknown opcode: " + name + " or incorrect number of arguments", line_number);
        return index+1;
    }
}


bool VM::readRegister(int id,int &value,int line_number){

    if (id >= (int)registers.size() ||  id < 0 ) {
         fail("out of range register id", line_number);
         return false; 
    }
    
    value= registers[id];
    return true;
}

void VM::writeRegister(int reg_id,int value,int line_number){


    if(reg_id<0){
        fail("out of range register id", line_number);
        return;
    }
    if (reg_id >= (int)registers.size() ) {
        registers.resize(reg_id + 1, 0);
    }
    registers[reg_id]=value;
}


void  VM:: decrementRegister(int reg_id,int line_number){

int value;
if(!readRegister(reg_id,value,line_number)) return;
//value can go negative
writeRegister(reg_id,value-1, line_number);


 }

 void VM:: incrementRegister(int reg_id,int line_number){

     
     int value;
     if(!readRegister(reg_id,value,line_number)) return;
    
     writeRegister(reg_id,value+1, line_number);

 }