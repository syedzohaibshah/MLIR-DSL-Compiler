#include "emunpu_device.h"

#include "../error.h"
#include <cstdint>
#include <string>
#include <iostream>




emunpuDevice::emunpuDevice(){

 emu_device=nullptr;
    
   emunpu_status_t status = emunpu_device_create(0,&emu_device);

   if (status!=EMUNPU_OK){
    
    emu_device=nullptr;
   }
    
}



   Error emunpuDevice::allocate(int size,int line,DeviceHandle& out_handle){
    if(emu_device == nullptr) return makeError("emunpu device not initialised", line);

    int alignment = limits().alignment_bytes;
    if(size%alignment!=0){
        
        return makeError("size must be a aligned to the device's alignment",line);
    }

       uint32_t num_floats=(uint32_t)(size/alignment); //size in bytes to number of floats

       emunpu_tensor_id_t tensor_id;
       
       emunpu_status_t status= emunpu_device_alloc(emu_device,
                                            EMUNPU_DT_FP32, /*rank */ 1,
                                            /*shape*/&num_floats, /*name*/nullptr,
                                            &tensor_id);
       if(status!=EMUNPU_OK)
       {

        return makeError("device allocation failed",line);
       }

       buffer_table.push_back({tensor_id, false, size});
       
       std::cout<<"[emunpu] alloc buffer "<<tensor_id<<" ("<<size<<" bytes)"<<std::endl;

       out_handle = (int)buffer_table.size() - 1; //VM only sees the 0‑based handle. 
       return ok();

   }


Error emunpuDevice::isValidHandle(DeviceHandle h, int line){
    if (h < 0 || h >= (int)buffer_table.size()) {
     
        return makeError("invalid buffer id",line);
    }
    if (buffer_table[h].freed) {
        return makeError("buffer already freed",line);
    }
    return ok();

   }


Error emunpuDevice::requireElements(DeviceHandle h, int64_t elements, const char* what, int line){
//  checks that buffer `h` holds at least `elements` fp32 values.
//   - Compute the needed bytes as int64_t: elements * sizeof(float). (D10)
//   - Reject elements < 0. Decide: is 0 a valid count? (see TODO 3)
//   - The error text must give the name `what`, the needed bytes and the buffer bytes.
//   - Read the buffer size from buffer_table[h].bytes.

if (elements < 0) {
    return makeError("negative element count for " + std::string(what), line);
}
int64_t needed_bytes = elements * sizeof(float);

//verify that each input/output buffer has enough memory to
// hold the number of elements required 
if (elements > buffer_table[h].bytes / (int64_t)sizeof(float)) {
    return makeError("buffer " + std::to_string(h) + " for " + std::string(what) + " needs " +
                      std::to_string(needed_bytes) + " bytes, but has only " + std::to_string(buffer_table[h].bytes), line);
}

    return ok();
}


    Error emunpuDevice::upload(DeviceHandle handle, const void* src, int size, int line){
        if(emu_device == nullptr) return makeError("emunpu device not initialised", line);

        RETURN_IF_ERROR(isValidHandle(handle, line)) ;
        
        emunpu_tensor_id_t tensor_id = buffer_table[handle].real_id;
        
        emunpu_status_t status =emunpu_device_upload(emu_device,tensor_id, src,size);

        if(status!=EMUNPU_OK)
        {
 
           
            return makeError("host to device upload failed", line);
        }

        
        std::cout<<"[emunpu] host to device copy with tensor_id: "<<tensor_id<<" ("<<size<<" bytes)"<<std::endl;

        return ok();
        
    }


     Error emunpuDevice::download(DeviceHandle handle, void* dst, int size, int line) {
         if(emu_device == nullptr) return makeError("emunpu device not initialised", line);

         RETURN_IF_ERROR(isValidHandle(handle, line));
                 emunpu_tensor_id_t tensor_id = buffer_table[handle].real_id;
         emunpu_status_t status= emunpu_device_download(emu_device,
                                                 tensor_id, dst,
                                                 size);
         if(status!=EMUNPU_OK)
         {
  
             return makeError("device to host download failed", line);
         }
         
         std::cout<<"[emunpu]  device to host  copy with tensor_id: "<<tensor_id<<"("<<size<<" bytes)"<<std::endl;

         return ok();
         
     }

      Error emunpuDevice::dispatch(const std::vector<int>& args, int line) {
          if(emu_device == nullptr) return makeError("emunpu device not initialised", line);

          if(args.size() < 2) {
            
              return makeError("LAUNCH expected at least kernel_id and num_buffers", line);
          }
      
     int op_id = args[0];
     int num_buffers = args[1];
 
     if(num_buffers < 0) { return makeError("LAUNCH has negative num_buffers", line); }

     int buffer_start = 2;
     size_t num_scalars_index = buffer_start + (size_t)num_buffers;
     if(args.size() <= num_scalars_index) { return makeError("LAUNCH missing dimensions or num_scalars", line); }
 
    int num_scalars = args[num_scalars_index];
    if(num_scalars < 0) { return makeError("LAUNCH has negative num_scalars", line); }

    int scalars_start = num_scalars_index + 1;

    int expected_total = scalars_start + num_scalars;
    
    if(args.size() != expected_total) { return makeError("LAUNCH arg count mismatch", line); }

    for(size_t i = buffer_start; i < num_scalars_index; ++i){
        RETURN_IF_ERROR(isValidHandle(args[i], line));
    }

    // Every scalar is a count or a dimension. A zero-size tensor cannot be
    // allocated (R1), so a zero here is almost always a lowering bug.
    for(size_t i = scalars_start; i < (size_t)expected_total; ++i){
        if(args[i] == 0) { return makeError("LAUNCH scalar must be greater than 0", line); }
    }

    std::cout<<"Launching kernel ID "<<op_id<<" with "<<num_buffers<<" buffers and "
        <<num_scalars<<" scalar args"<<std::endl;

    if (op_id == 0) {
        // MATMUL
 
      return do_matmul(num_buffers,num_scalars, scalars_start,buffer_start,args,line);

  

    } else if (op_id == 1) {
        // VEC_ADD
        
        return do_vec_add(num_buffers,num_scalars, scalars_start,
            buffer_start,args,line);

    } else if (op_id == 2) {
        // RELU
        return do_relu(num_buffers,num_scalars, scalars_start,
            buffer_start,args,line);

    } else if (op_id == 3) {
        // BIAS_ADD
        return do_bais_add(num_buffers,num_scalars, scalars_start,
            buffer_start,args,line);
    } else {
        return makeError("unknown op_id", line);
    }
    return ok();
 
      }

        Error emunpuDevice::free(DeviceHandle handle,int line){
           if(emu_device == nullptr) return makeError("emunpu device not initialised", line);

            RETURN_IF_ERROR(isValidHandle(handle, line));

             emunpu_tensor_id_t tensor_id=buffer_table[handle].real_id;
          
            emunpu_status_t status= emunpu_device_free(emu_device,tensor_id);

            if(status!=EMUNPU_OK)
            {
              return makeError("device free failed", line);
            }

            buffer_table[handle].freed = true;
  std::cout<<"[emunpu] freed buffer with tensor id:"<<tensor_id<<std::endl;

  return ok();
            
        }

    Error emunpuDevice::do_matmul(int num_buffers,int num_scalars,int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 3) {
            return makeError("wrong number of buffers/scalars for MATMUL", line);
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t M = args[scalars_start+0];
        uint32_t N = args[scalars_start+1];
        uint32_t K = args[scalars_start+2];

        //  check the three buffers against the shape, before the emunpu call.
        //   A holds M*K values, B holds K*N, C holds M*N. Use requireElements.
        //   Compute each product as int64_t, so large Llama shapes do not overflow.
        //   Pass args[buffer_start+i] as the handle. Use the handle, not the real id.
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)M * K, "MATMUL A", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)K * N, "MATMUL B", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)M * N, "MATMUL C", line));
        emunpu_status_t status = emunpu_matmul(emu_device, nullptr, c, a, b, M, N, K, 0);
        if (status != EMUNPU_OK) {
        
            return makeError("emunpu_matmul failed", line);
        }

        
        return ok();
    }

    Error emunpuDevice::do_vec_add(int num_buffers,int num_scalars,int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 1) {
            return makeError("wrong number of buffers/scalars for VEC_ADD", line);
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t count = args[scalars_start+0];


        //  check a, b and c each hold `count` values (requireElements).
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "VEC_ADD A", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)count, "VEC_ADD B", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)count, "VEC_ADD C", line));
        emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, b, count, EMUNPU_EW_ADD, 0);
        if (status != EMUNPU_OK) {
            return makeError("emunpu_vector_op failed", line);
        }
        return ok();
    }

    Error emunpuDevice::do_relu(int num_buffers,int num_scalars,
        int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 2 || num_scalars != 1) {
            return makeError("wrong number of buffers/scalars for RELU", line);
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+1]].real_id;
        uint32_t count = args[scalars_start+0];

        //  check a (input) and c (output) each hold `count` values (requireElements).
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "RELU input", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)count, "RELU output", line));
        emunpu_status_t status = emunpu_activation(emu_device, nullptr, c, a, count, EMUNPU_ACT_RELU, 0, 0);
        if (status != EMUNPU_OK) {
            return makeError("emunpu_activation failed", line);
        }
        return ok();
    }

    Error emunpuDevice::do_bais_add(int num_buffers,int num_scalars,
        int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 2) {
            return makeError("wrong number of buffers/scalars for BIAS_ADD", line);
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t bias = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t count = args[scalars_start+0];
        uint32_t out_features = args[scalars_start+1];


        //  check a and c each hold `count` values. Check bias holds out_features values.
        //   Also check out_features > 0 and out_features <= the bias element count.
        //   Decide: must count be a multiple of out_features? Ask the compiler team
        //   what the bias period means before you write this rule.
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "BIAS_ADD input", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)out_features, "BIAS_ADD bias", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)count, "BIAS_ADD output", line));
        if (out_features > count) {
            return makeError("BIAS_ADD out_features cannot exceed count", line);
        }
        emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, bias, count, EMUNPU_EW_ADD, out_features);
        if (status != EMUNPU_OK) {
            return makeError("emunpu_vector_op failed", line);
        }
        return ok();
    }


       
std::string emunpuDevice::name() const {


                 return "emunpu";
             }

DeviceLimits emunpuDevice::limits() const {
    DeviceLimits limits;
     limits.alignment_bytes = 4; // default alignment for emunpu device

   emunpu_global_mem_t* global_mem = (emu_device!=nullptr)? emunpu_device_global_mem(emu_device):nullptr;



if( global_mem == nullptr) {
        limits.max_allocation_bytes = 0;
        limits.total_memory_bytes = 0;
        limits.max_live_buffers = 0;
        return limits;
    }

    limits.max_allocation_bytes = global_mem->capacity; 
    limits.total_memory_bytes = global_mem->capacity;  
    limits.max_live_buffers = 1024;             //  max tensor on emunpu
    return limits;
}

Error emunpuDevice::status() const {
    if(emu_device == nullptr) return makeError("emunpu device was not created", 0);
    return ok();
}

emunpuDevice::~emunpuDevice() {
    if (emu_device) emunpu_device_destroy(emu_device);
}





