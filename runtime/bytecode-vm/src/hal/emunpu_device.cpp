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

    if(size%4!=0){
        
        return makeError("size must be a multiple of 4 bytes",line);
    }

       uint32_t num_floats=(uint32_t)(size/4);

       emunpu_tensor_id_t tensor_id;
       
       emunpu_status_t status= emunpu_device_alloc(emu_device,
                                            EMUNPU_DT_FP32, /*rank */ 1,
                                            /*shape*/&num_floats, /*name*/nullptr,
                                            &tensor_id);
       if(status!=EMUNPU_OK)
       {

        return makeError("device allocation failed",line);
       }

       buffer_table.push_back({tensor_id,false});
       
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
        emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, bias, count, EMUNPU_EW_ADD, out_features);
        if (status != EMUNPU_OK) {
            return makeError("emunpu_vector_op failed", line);
        }
        return ok();
    }


       
std::string emunpuDevice::name() const {


                 return "emunpu";
             }

             
Error emunpuDevice::status() const {
    if(emu_device == nullptr) return makeError("emunpu device was not created", 0);
    return ok();
}

emunpuDevice::~emunpuDevice() {
    if (emu_device) emunpu_device_destroy(emu_device);
}





