#include "vm_core.h"
#include "error.h"
#include "verifier.h"   
#include <iostream>




VM::VM(Device* device, RuntimeLimits limits)
    : runtime_limits(limits),       // 1st declared
      io(runtime_limits),           // 2nd declared — uses the member above
      device(device)                // 3rd declared
{}


void VM::setLeakPolicy(LeakPolicy policy){
    leak_policy = policy;
}

Error VM::bindSignature(const Bytecode& bytecode){

    if (signature_bound) {
        return makeError("bindSignature called twice", 0);
    }

    bound_inputs = bytecode.inputs();

    bound_outputs = bytecode.outputs();

    signature_bound = true;
    return ok();
}

//  the host gives an input. Check it against the signature first.
Error VM::setInput(int slot, void* data, int size){
    RETURN_IF_ERROR(checkDeclared(bound_inputs, "input", slot, size, 0));
    return io.setInput(slot, data, size);
}

//the host reads an output. Same rules as setInput, on bound_outputs.
Error VM::getOutput(int slot, void* out_ptr, int size){
    RETURN_IF_ERROR(checkDeclared(bound_outputs, "output", slot, size, 0));
    return io.getOutput(slot, out_ptr, size);
}

// checks the declared slots in header aginst the input/output slots and upload/download
// that were actually used in the run. 
// If any declared slot was not used, return an error. then size are checked 
Error VM::checkDeclared(const std::map<int, SlotDecl>& decls, const std::string& kind,
                        int slot, int size, int line) const {
    if (!signature_bound) {
        return makeError("bindSignature must be called before using " + kind + " slots", line);
    }
    auto it = decls.find(slot);
    if (it == decls.end()) {
        return makeError(kind + " slot " + std::to_string(slot) + " is not declared in the signature", line);
    }
    const SlotDecl& decl = it->second;
    if (size != decl.bytes) {
        return makeError(kind + " slot " + std::to_string(slot) + " (" + decl.name + ") expects " +
                         std::to_string(decl.bytes) + " bytes, but got " + std::to_string(size), line);
    }
    return ok();
}

Error VM::execute(const std::vector<opcode>& opcodes){
     int next_index;
    int opcode_size=opcodes.size();

    // Step limit: stops a runaway loop (for example, a wrong COND_BR that
    // jumps back forever). A real run re-runs instructions many times, so the
    // ceiling is large.
    long long executed = 0;
    const long long step_limit = runtime_limits.STEP_LIMIT; // 1e8

       if (!signature_bound) {
        return makeError("bindSignature must be called before execute()", 0);
    }

    // check the whole program before the first device call.(see verifer.cpp)

    const DeviceLimits device_limits = device->limits();
    VerifyContext ctx{runtime_limits, device_limits, bound_inputs, bound_outputs};
    RETURN_IF_ERROR(verifyProgram(opcodes, ctx));

    //  before the first instruction, every declared input must be set.
    //  loop over bound_inputs. For each slot, ask io whether it was set.  if a slot is not set, return an error 
    for (const auto& input : bound_inputs) {
        int slot = input.first;
        const SlotDecl& decl = input.second;
        Error check_input_error = io.checkInputSlot(slot, decl.bytes, 0);
        if (!check_input_error.ok) {
            return makeError("input slot " + std::to_string(slot) + " (" + decl.name + ") is not set before execution", 0);
        }
    }
  

 

    //  start each run with no used slots.
    used_inputs.clear();
    used_outputs.clear();

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
    RETURN_IF_ERROR(checkDeclaredSlotsUsed());
    RETURN_IF_ERROR(handleLiveBuffers());
    return  ok();
}

// COPY_H2D buffer_id input_slot size
Error VM::copyHostToDevice(int desti, int src_slot, int size, int line_number){
    RETURN_IF_ERROR(checkSize(size, line_number)); //check if the size is greater than 0
    RETURN_IF_ERROR(checkDeclared(bound_inputs, "input", src_slot, size, line_number));

    //check if the buffer we are copying to is allocated and live
    int buffer_size;
    RETURN_IF_ERROR(checkBuffer(desti, buffer_size, line_number)); 

    if(size>buffer_size){
        return makeError("COPY_H2D size "+std::to_string(size)+" exceeds buffer size "+std::to_string(buffer_size), line_number);
    }

 
   
    RETURN_IF_ERROR(device->upload(desti, io.inputData(src_slot), size, line_number));

    // Record the slot only after the upload succeeds (step 7).
    used_inputs.insert(src_slot);

    return ok();
}

// COPY_D2H output_slot buffer_id size
Error VM::copyDeviceToHost(int dest_slot, int src, int size, int line_number) {

    RETURN_IF_ERROR(checkSize(size, line_number)); //check if the size is greater than 0

    // Before prepareOutputSlot, which resizes the slot (a side effect).
    RETURN_IF_ERROR(checkDeclared(bound_outputs, "output", dest_slot, size, line_number));

    int buffer_size;
    RETURN_IF_ERROR(checkBuffer(src, buffer_size, line_number)); //check if the buffer
    if(size>buffer_size){
        return makeError("COPY_D2H size "+std::to_string(size)+" exceeds buffer size "+std::to_string(buffer_size), line_number);
    }

    uint8_t* out_buf = io.prepareOutputSlot(dest_slot, size); //resize output slot and return the pointer to it


     if(out_buf == nullptr) {
        return makeError(" output slot is out of bounds", line_number);
    }

    RETURN_IF_ERROR(device->download(src, out_buf, size, line_number));
    // Record the slot only after the download succeeds (step 7).
    used_outputs.insert(dest_slot);

    io.markOutputSlotWritten(dest_slot);
    return ok();
}


// Buffer table: the VM's own record of every buffer, so device-independent
// checks run once for every backend. See doc/validation-design.md section 7.

Error VM::allocateBuffer(int size, int line_number){

    // The stricter of the two caps wins. A device reporting 0 has no limit.
    int device_cap = device->limits().max_live_buffers;
    int cap = runtime_limits.MAX_LIVE_BUFFERS;
    if(device_cap > 0 && device_cap < cap){
        cap = device_cap;
    }
    if(live_buffer_count >= cap){
        return makeError("too many live buffers (limit " + std::to_string(cap) +
                         ") - a buffer is probably never freed", line_number);
    }

    DeviceHandle handle;
    RETURN_IF_ERROR(device->allocate(size, line_number, handle));

    // Verify the append-only contract instead of assuming it.
    if(handle != (int)buffers.size()){
        return makeError("device returned handle " + std::to_string(handle) +
                         ", expected " + std::to_string(buffers.size()) +
                         " - Device must hand out handles in order and never reuse them",
                         line_number);
    }

    buffers.push_back(BufferSlot{size, true, line_number});
    live_buffer_count++;
    return ok();
}

Error VM::freeBuffer(int handle, int line_number){

    int size_unused; //freed buuffer size
    RETURN_IF_ERROR(checkBuffer(handle, size_unused, line_number));

    RETURN_IF_ERROR(device->free(handle, line_number));

    // Mark dead, never erase: the index must keep matching the handle.
    buffers[handle].live = false;
    live_buffer_count--;
    return ok();
}

Error VM::handleLiveBuffers(){
    for(int handle = 0; handle < (int)buffers.size(); ++handle){
        const BufferSlot& slot = buffers[handle];
        if(!slot.live) continue;

        if(leak_policy == LeakPolicy::STRICT){
            return makeError("buffer " + std::to_string(handle) + " (" + std::to_string(slot.size) +
                             " bytes) was allocated here and never freed", slot.alloc_line);
        }
        std::cerr << "Warning: buffer " << handle << " (" << slot.size
                  << " bytes) was never freed; freeing it at the end of the run" << std::endl;
        RETURN_IF_ERROR(freeBuffer(handle, slot.alloc_line));
    }
    return ok();
}

Error VM::checkBuffer(int handle, int& out_size, int line_number) const{

    if(handle < 0 || handle >= (int)buffers.size()){
        return makeError("buffer " + std::to_string(handle) + " was never allocated",
                         line_number);
    }

    //  double-frees
    if(!buffers[handle].live){
        return makeError("buffer " + std::to_string(handle) + " was already freed",
                         line_number);
    }
    out_size = buffers[handle].size;
    return ok();
}

// REG_ID and TARGET arguments reach here only as literals: verifyProgram
// rejects `@` on them before execute() starts the dispatch loop (rule R8).
Error VM::dispatch(const std::string& name, const std::vector<Arg>& args, int line_number,int index,int opcode_size, int& next_index){


    //value
    if(name=="ALLOC" && args.size()==1){

           int alloc_size;
           RETURN_IF_ERROR(resolveValue(args[0], alloc_size, line_number));

           RETURN_IF_ERROR(checkSize(alloc_size, line_number)); //check if the size is greater than 0

        RETURN_IF_ERROR(allocateBuffer(alloc_size, line_number));

         next_index = index+1;
         return ok();
    }
    else if(name=="FREE" && args.size()==1){ //value
            int buff_id;
           RETURN_IF_ERROR(resolveValue(args[0], buff_id, line_number));

        RETURN_IF_ERROR(freeBuffer(buff_id, line_number));
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
            if(arg_value<0){
                return makeError("LAUNCH argument cannot be negative", line_number);
            }

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

       RETURN_IF_ERROR(writeRegister(args[0].value,set_value,line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="DEC" && args.size()==1){ //id

        RETURN_IF_ERROR(decrementRegister(args[0].value,line_number));
        next_index = index+1;
        return ok();
    }
    else if(name=="INC" && args.size()==1){//id

        RETURN_IF_ERROR(incrementRegister(args[0].value,line_number));
        next_index = index+1;
        return ok();
    }
    else if (name=="BR" && args.size()==1){
        //just jump to the target_index

        next_index = args[0].value;
        //out of bounds check
        if(next_index<0 || next_index>=opcode_size){
            return makeError("BR target index out of bounds", line_number);
        }
        return ok();
    }
    else if(name=="COND_BR" && args.size()==3){//ids

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

    if (reg_id >= runtime_limits.MAX_REGISTERS ||   reg_id < 0 ) {

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




    if(reg_id<0 || reg_id >= runtime_limits.MAX_REGISTERS){
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



//  called at the end of a run. Every declared slot must have been used.
Error VM::checkDeclaredSlotsUsed() const {
    //  walk bound_inputs. A slot that is not in used_inputs is an error.
    for (const auto& pair : bound_inputs) {
        int slot = pair.first;
        const SlotDecl& decl = pair.second;
        if (used_inputs.find(slot) == used_inputs.end()) {
            return makeError("input slot " + std::to_string(slot) + " (" + decl.name + ") was declared but never used", 0);
        }
    }
    //  walk bound_outputs. A slot that is not in used_outputs is an error.
    for (const auto& pair : bound_outputs) {
        int slot = pair.first;
        const SlotDecl& decl = pair.second;
        if (used_outputs.find(slot) == used_outputs.end()) {
            return makeError("output slot " + std::to_string(slot) + " (" + decl.name + ") was declared but never used", 0);
        }
    }

    return ok();
}

Error VM::checkSize(int size, int line) const {
    if (size <= 0) {
        return makeError("size must be greater than 0", line);
    }
    return ok();
}
