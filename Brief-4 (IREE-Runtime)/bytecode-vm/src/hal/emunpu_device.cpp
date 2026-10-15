#include "emunpu_device.h"

#include "../error.h"
#include <cstdint>
#include <string>
#include <iostream>




emunpuDevice::emunpuDevice(){

 emu_device=nullptr;
    
   emunpu_status_t status = emunpu_device_create(0,&emu_device);

   if (status!=EMUNPU_OK){

       fail("device creation failed", 0);
       emu_device=nullptr;
   }
    
}


   DeviceHandle emunpuDevice::allocate(int size,int line){

       uint32_t num_floats=(uint32_t)(size/4);

       emunpu_tensor_id_t tensor_id;
       
       emunpu_status_t status= emunpu_device_alloc(emu_device,
                                            EMUNPU_DT_FP32, /*rank */ 1,
                                            /*shape*/&num_floats, /*name*/nullptr,
                                            &tensor_id);
       if(status!=EMUNPU_OK)
       {

           fail("device mem allocation failed", line);
           return -1;
       }

       buffer_table.push_back({tensor_id,false});
       
       std::cout<<"[emunpu] alloc buffer "<<tensor_id<<" ("<<size<<" bytes)"<<std::endl;

       return (int)buffer_table.size() - 1; //VM only sees the 0‑based handle. 

   }


bool emunpuDevice::isValidHandle(DeviceHandle h, int line){
    if (h < 0 || h >= (int)buffer_table.size()) {
        fail("invalid buffer id", line);
        return false;
    }
    if (buffer_table[h].freed) {
        fail("buffer already freed", line);
        return false;
    }
    return true;

   }


    bool emunpuDevice::upload(DeviceHandle handle, const void* src, int size, int line){

        if (!isValidHandle(handle, line)) return false;
        
        emunpu_tensor_id_t tensor_id = buffer_table[handle].real_id;
        
        emunpu_status_t status =emunpu_device_upload(emu_device,tensor_id, src,size);

        if(status!=EMUNPU_OK)
        {
 
            fail("host to device upload failed", line);
            return false;
        }

        
        std::cout<<"[emunpu] host to device copy with tensor_id: "<<tensor_id<<" ("<<size<<" bytes)"<<std::endl;

        return true;
        
    }


     bool emunpuDevice::download(DeviceHandle handle, void* dst, int size, int line) {

         if (!isValidHandle(handle, line)) return false;
         
                 emunpu_tensor_id_t tensor_id = buffer_table[handle].real_id;
         emunpu_status_t status= emunpu_device_download(emu_device,
                                                 tensor_id, dst,
                                                 size);
         if(status!=EMUNPU_OK)
         {
  
             fail("device to host download failed", line);
             return false;
         }
         
         std::cout<<"[emunpu]  device to host  copy with tensor_id: "<<tensor_id<<"("<<size<<" bytes)"<<std::endl;

    return true;
         
     }

      bool emunpuDevice::dispatch(const std::vector<int>& args, int line) {

          if(args.size() < 2) {
              fail("LAUNCH expected at least kernel_id and num_buffers", line); return false;
          }
      
     int op_id = args[0];
     int num_buffers = args[1];
 
     if(num_buffers < 0) { fail("LAUNCH has negative num_buffers", line); return false; }

     int buffer_start = 2;
     size_t num_scalars_index = buffer_start + (size_t)num_buffers;
     if(args.size() <= num_scalars_index) { fail("LAUNCH missing dimensions or num_scalars", line); return false; }
 
    int num_scalars = args[num_scalars_index];
    if(num_scalars < 0) { fail("LAUNCH has negative num_scalars", line); return false; }

    int scalars_start = num_scalars_index + 1;

    int expected_total = scalars_start + num_scalars;
    
    if(args.size() != expected_total) { fail("LAUNCH arg count mismatch", line); return false; }

    for(size_t i = buffer_start; i < num_scalars_index; ++i){
        if(!isValidHandle(args[i], line)) return false;
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
        fail("unknown op_id", line);
        return false;
    }
    return true;
 
      }

        void emunpuDevice::free(DeviceHandle handle,int line){

            if (!isValidHandle(handle, line)) return; 

             emunpu_tensor_id_t tensor_id=buffer_table[handle].real_id;
          
            emunpu_status_t status= emunpu_device_free(emu_device,tensor_id);

            if(status!=EMUNPU_OK)
            {
                fail("device mem de-allocation failed", line);
                return;
            }

            buffer_table[handle].freed = true;
  std::cout<<"[emunpu] freed buffer with tensor id:"<<tensor_id<<std::endl;
            
        }

    bool emunpuDevice::do_matmul(int num_buffers,int num_scalars,int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 3) {
            fail("wrong number of buffers/scalars for MATMUL", line);
            return false;
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t M = args[scalars_start+0];
        uint32_t N = args[scalars_start+1];
        uint32_t K = args[scalars_start+2];
        emunpu_status_t status = emunpu_matmul(emu_device, nullptr, c, a, b, M, N, K, 0);
        if (status != EMUNPU_OK) {
            fail("emunpu_matmul failed", line);
            return false;
        }

        
        return true;
    }

    bool emunpuDevice::do_vec_add(int num_buffers,int num_scalars,int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 1) {
            fail("wrong number of buffers/scalars for VEC_ADD", line);
            return false;
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t count = args[scalars_start+0];
        emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, b, count, EMUNPU_EW_ADD, 0);
        if (status != EMUNPU_OK) {
            fail("emunpu_vector_op failed", line);
            return false;
        }
        return true;
    }

    bool emunpuDevice::do_relu(int num_buffers,int num_scalars,
        int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 2 || num_scalars != 1) {
            fail("wrong number of buffers/scalars for RELU", line);
            return false;
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+1]].real_id;
        uint32_t count = args[scalars_start+0];
        emunpu_status_t status = emunpu_activation(emu_device, nullptr, c, a, count, EMUNPU_ACT_RELU, 0, 0);
        if (status != EMUNPU_OK) {
            fail("emunpu_activation failed", line);
            return false;
        }
        return true;
    }

    bool emunpuDevice::do_bais_add(int num_buffers,int num_scalars,
        int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        if (num_buffers != 3 || num_scalars != 2) {
            fail("wrong number of buffers/scalars for BIAS_ADD", line);
            return false;
        }
        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t bias = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t count = args[scalars_start+0];
        uint32_t out_features = args[scalars_start+1];
        emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, bias, count, EMUNPU_EW_ADD, out_features);
        if (status != EMUNPU_OK) {
            fail("emunpu_vector_op failed", line);
            return false;
        }
        return true;
    }


       
std::string emunpuDevice::name() const {


                 return "emunpu";
             }

             
emunpuDevice::~emunpuDevice() {
    if (emu_device) emunpu_device_destroy(emu_device);
}





