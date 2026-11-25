#include "emunpu_device.h"

#include "../error.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <iostream>


//// Scalar Arguments specs

////////// Scalar argeuments are validated against these specs, which are per-kernel. The
////////// specs are used by checkScalar to validate the range of each scalar.

static const ScalarSpec MATMUL_SCALARS[]   = {{ScalarKind::DIM,0}, {ScalarKind::DIM,0}, {ScalarKind::DIM,0}};
static const ScalarSpec VEC_ADD_SCALARS[]  = {{ScalarKind::DIM,0}};
static const ScalarSpec RELU_SCALARS[]     = {{ScalarKind::DIM,0}};
static const ScalarSpec BIAS_ADD_SCALARS[] = {{ScalarKind::DIM,0}, {ScalarKind::DIM,0}};
static const ScalarSpec RMS_NORM_SCALARS[] = {{ScalarKind::DIM,0}, {ScalarKind::DIM,0}};
static const ScalarSpec SOFTMAX_SCALARS[]  = {{ScalarKind::DIM,0}, {ScalarKind::DIM,0}, {ScalarKind::FLAG,0}};
static const ScalarSpec TRANSPOSE_SCALARS[] = {{ScalarKind::DIM,0}, {ScalarKind::DIM,0}};




// The activation kernel has a fixed set of kinds, so it uses an ENUM scalar to
// select one. The enum_count is the number of kinds, so checkScalar can validate
// the scalar value against that range.
// EMUNPU_ACT_COUNT (arch.h) is 24: kinds 0 (RELU) .. 23 (ERF).
static const int EMUNPU_ACTIVATION_KIND_COUNT = 24;
// alpha/beta always ride along as FLOAT_BITS, even for kinds that ignore them
// (callers pass the bits of 0.0f) -- ACTIVATION_SCALARS's length cannot vary
// by kind, since KernelSignature declares one fixed num_scalars per op_id.
static const ScalarSpec ACTIVATION_SCALARS[] = {
    {ScalarKind::DIM,0},                                   // count
    {ScalarKind::ENUM, EMUNPU_ACTIVATION_KIND_COUNT},       // kind
    {ScalarKind::FLOAT_BITS,0},                             // alpha (LEAKY_RELU slope, CLAMP lower bound)
    {ScalarKind::FLOAT_BITS,0},                             // beta  (CLAMP upper bound)
};

static const int EMUNPU_EW_KIND_COUNT=9;
// bcast_period: PARAM, not DIM -- 0 is emunpu_vector_op's documented meaning
// for "no broadcast, equal shape" (device.h), i.e. the common case, not an
// edge case. DIM's `> 0` rule would reject the normal value
static const ScalarSpec VECTOR_OP_SCALARS[] = {
    {ScalarKind::DIM,0},                                   // count
    {ScalarKind::ENUM, EMUNPU_EW_KIND_COUNT},               // kind
    {ScalarKind::PARAM,0},                                 // broadcast period
};

static const int  EMUNPU_REDUCE_KIND_COUNT=7;

static const ScalarSpec REDUCE_OP_SCALARS[] = {
    {ScalarKind::DIM,0}, //rows
    {ScalarKind::DIM,0},        //width                            
    {ScalarKind::ENUM, EMUNPU_REDUCE_KIND_COUNT},               // kind
                     
};

static const ScalarSpec MATVEC_OP_SCALARS[] = {
    {ScalarKind::DIM,0}, //M
    {ScalarKind::DIM,0},        //K                                         
};


static const ScalarSpec FMA_OP_SCALARS[] = {
    {ScalarKind::DIM,0}, //count
    };

static const ScalarSpec GEMM_OP_SCALARS[] = {
    {ScalarKind::DIM,0}, //M
    {ScalarKind::DIM,0}, //K
    {ScalarKind::DIM,0}, //N
    {ScalarKind::FLAG,0}, //transpose_b
    {ScalarKind::FLAG,0}, //fuse_relu
    };





////////////// @brief A table of all available kernel signatures.
static const KernelSignature KERNEL_TABLE[] = {
    {"MATMUL",     3, 3, MATMUL_SCALARS,     3},
    {"VEC_ADD",    3, 3, VEC_ADD_SCALARS,    1},
    {"RELU",       2, 2, RELU_SCALARS,       1},   // scalars: count only. dispatch forwards to do_activation(kind=RELU)
    {"BIAS_ADD",   3, 3, BIAS_ADD_SCALARS,   2},
    {"RMS_NORM",   2, 3, RMS_NORM_SCALARS,   2},   // 2 buffers skips gamma, 3 applies it
    {"SOFTMAX",    2, 2, SOFTMAX_SCALARS,    3},
    {"TRANSPOSE",  2, 2, TRANSPOSE_SCALARS, 2},
    {"ACTIVATION", 2, 2, ACTIVATION_SCALARS, 4},   // op_id 7. count, kind, alpha_bits, beta_bits
    // 2 buffers (a, c) for a unary kind (RELU, COPY); 3 (a, b, c) for a binary
    // kind.
    {"VECTOR_OP",  2, 3, VECTOR_OP_SCALARS, 3},   // op_id 8. count, kind, broadcast period
    {"REDUCE_OP",  2, 2, REDUCE_OP_SCALARS, 3},   // op_id 9. rows, width, kind
    {"MATVEC_OP",  3, 3, MATVEC_OP_SCALARS, 2},   // op_id 10. M, K
    {"FMA_OP",     3, 3, FMA_OP_SCALARS, 1},      // op_id 11. count
    {"GEMM_OP",    3, 4, GEMM_OP_SCALARS, 5},     // op_id 12. M, K, N, transpose_b, fuse_relu
};

static const int KERNEL_TABLE_SIZE = (int)(sizeof(KERNEL_TABLE) / sizeof(KERNEL_TABLE[0]));

const KernelSignature* lookupKernelSignature(int op_id){
    if(op_id < 0 || op_id >= KERNEL_TABLE_SIZE) return nullptr;
    return &KERNEL_TABLE[op_id];
}

// Checks one scalar against the range its kind allows.
static Error checkScalar(const ScalarSpec& spec, int value, const char* kernel_name,
                         int position, int line){
    std::string where = std::string(kernel_name) + " scalar " + std::to_string(position);
    switch(spec.kind){
        case ScalarKind::DIM:
            if(value <= 0) return makeError(where + " is a dimension and must be greater than 0", line);
            break;
        case ScalarKind::FLAG:
            if(value != 0 && value != 1) return makeError(where + " is a flag and must be 0 or 1", line);
            break;
        case ScalarKind::PARAM:
            break;   // any value is valid, including negative
        case ScalarKind::ENUM:
            if(value < 0 || value >= spec.enum_count){
                return makeError(where + " is an enum and must be 0 to " +
                                  std::to_string(spec.enum_count - 1), line);
            }
            break;
        case ScalarKind::FLOAT_BITS:
            // all float bits are valid, including negative. The VM passes the bits of a float.
            break;
    }
    return ok();
}


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
 
     //no of buffer must be non negative
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

    // The signature row decides the counts and every scalar range, so a new
    // kernel never needs a special case here.
    const KernelSignature* sig = lookupKernelSignature(op_id);
    if(sig == nullptr) return makeError("unknown op_id", line);

    if(num_buffers < sig->min_buffers || num_buffers > sig->max_buffers){
        return makeError(std::string("wrong number of buffers for ") + sig->name, line);
    }
    if(num_scalars != sig->num_scalars){
        return makeError(std::string("wrong number of scalars for ") + sig->name, line);
    }
    // Check each scalar against the range its kind allows. The signature row has 
    //one entry per scalar position, in order.
    for(int i = 0; i < sig->num_scalars; ++i){
        RETURN_IF_ERROR(checkScalar(sig->scalars[i], args[scalars_start + i], sig->name, i, line));
    }

    std::cout<<"Launching kernel ID "<<op_id<<" with "<<num_buffers<<" buffers and "
        <<num_scalars<<" scalar args"<<std::endl;

    if (op_id == 0) {
        // MATMUL
 
      return do_matmul(scalars_start,buffer_start,args,line);

  

    } else if (op_id == 1) {
        // VEC_ADD
        
        return do_vec_add(scalars_start,buffer_start,args,line);

    } else if (op_id == 2) {
        // RELU — forwards to the same emunpu_activation call ACTIVATION uses,
        // with kind fixed. See runActivation.
        return do_relu(scalars_start,buffer_start,args,line);

    } else if (op_id == 3) {
        // BIAS_ADD
        return do_bais_add(scalars_start,buffer_start,args,line);
    }else if(op_id==4){
        // RMS_NORM
        return do_rms_norm(num_buffers, scalars_start, buffer_start, args, line);
    } else if(op_id==5){

        return do_softmax(scalars_start,buffer_start,args,line);


    }else if(op_id==6){

        return do_transpose(scalars_start,buffer_start,args,line);

    }else if(op_id==7){
      
        return do_activation(scalars_start,buffer_start,args,line);

    }
    else if(op_id==8){

        return do_vector_op(num_buffers,scalars_start,buffer_start,args,line);


    }
    else if(op_id==9){

        return do_reduce_op(num_buffers,scalars_start,buffer_start,args,line);
    }
    else if(op_id==10){

        return  do_matvec( num_buffers,scalars_start, buffer_start,args,line);

    }
    else if(op_id==11){

        return do_fma(num_buffers,scalars_start, buffer_start,args,line);
    }else if(op_id==12){

        return  do_gemm(num_buffers,scalars_start, buffer_start,args,line);
    }
    else {
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

    Error emunpuDevice::do_matmul(int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

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
        emunpu_status_t status = emunpu_matmul(emu_device, nullptr, c, a, b, M, N, K, 0);  //se what we have to do of 0(transpose)
        if (status != EMUNPU_OK) {
        
            return makeError("emunpu_matmul failed", line);
        }

        
        return ok();
    }

    Error emunpuDevice::do_vec_add(int scalars_start,
        int buffer_start,const std::vector<int>& args,int line){

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

    Error emunpuDevice::do_bais_add(int scalars_start,int buffer_start,const std::vector<int>& args,int line){

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


    Error emunpuDevice::do_rms_norm(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        const double eps = 1e-5;
        bool has_gamma = (num_buffers == 3);
        uint32_t rows = args[scalars_start+0];
        uint32_t width = args[scalars_start+1];

        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)rows*width, "RMS_NORM input", line));

        emunpu_tensor_id_t gamma = 0;   // 0 tells emunpu "no gamma" (device.c: if (gamma))
        emunpu_tensor_id_t c;

        if(has_gamma) {
            gamma = buffer_table[args[buffer_start+1]].real_id;
            c = buffer_table[args[buffer_start+2]].real_id;
            RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)width, "RMS_NORM gamma", line));
            RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)rows*width, "RMS_NORM output", line));
        } else {
            c = buffer_table[args[buffer_start+1]].real_id;
            RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)rows*width, "RMS_NORM output", line));
        }

        emunpu_status_t status = emunpu_rmsnorm(emu_device, nullptr, c, a, gamma, rows, width, eps);
        if(status != EMUNPU_OK){
            return makeError("emunpu_rmsnorm failed", line);
        }

        return ok();
    }

    Error emunpuDevice:: do_softmax(int scalars_start,int buffer_start,const std::vector<int>& args,int line){
            emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
            emunpu_tensor_id_t c = buffer_table[args[buffer_start+1]].real_id;

            uint32_t rows = args[scalars_start+0];
            uint32_t width = args[scalars_start+1];
            int log_softmax = args[scalars_start+2]; 

            RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)rows*width, "SOFTMAX input", line));
            RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)rows*width, "SOFTMAX output", line));

        //log_softmax is a boolean flag, but we pass it as an int to the emunpu_softmax function.
        // The emunpu_softmax function will interpret any non-zero value as true, and zero as false.
emunpu_status_t status= emunpu_softmax(emu_device, nullptr,
                               c,  a, rows,width, int(log_softmax));

                               if(status != EMUNPU_OK){
                                return makeError("emunpu_softmax failed", line);
                               }
                               return ok();


        }

       Error emunpuDevice::do_transpose(int scalars_start,int buffer_start,const std::vector<int>& args,int line){

        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+1]].real_id;

        uint32_t rows = args[scalars_start+0];
        uint32_t cols = args[scalars_start+1];

        // rows and cols describe the INPUT. The output is [cols, rows], so it
        // holds the same number of elements. The two counts are equal on
        // purpose; only the shape differs.
        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)rows*cols, "TRANSPOSE input", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)rows*cols, "TRANSPOSE output", line));



emunpu_status_t status=  emunpu_transpose(emu_device,
                                nullptr,  c,  a,rows,  cols);


        if(status != EMUNPU_OK){
        return makeError("emunpu_transpose failed", line);
         }
        return ok();


       }


    // Shared body for RELU and ACTIVATION. Neither reads `kind` differently —
    // RELU just never lets the caller choose it.
    Error emunpuDevice::runActivation(int buffer_start,const std::vector<int>& args,
                                      uint32_t count,emunpu_act_kind_t kind,int line,double alpha,double beta){

        emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+1]].real_id;

        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "ACTIVATION input", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)count, "ACTIVATION output", line));

      
        emunpu_status_t status = emunpu_activation(emu_device, nullptr, c, a, count, kind, alpha, beta);
        if (status != EMUNPU_OK) {
            return makeError("emunpu_activation failed", line);
        }

        return ok();
    }

    // RELU (op_id 2). LAUNCH only carries `count`; `kind` is fixed to RELU,
    // not read from args — RELU's signature row has no kind scalar at all.
    Error emunpuDevice::do_relu(int scalars_start,int buffer_start,const std::vector<int>& args,int line){
        uint32_t count = args[scalars_start+0];
        return runActivation(buffer_start, args, count, EMUNPU_ACT_RELU, line);
    }

    Error emunpuDevice::do_activation(int scalars_start,int buffer_start,const std::vector<int>& args,int line){
        uint32_t count = args[scalars_start+0];
        emunpu_act_kind_t kind = (emunpu_act_kind_t)args[scalars_start+1];

        // alpha/beta are always present (ACTIVATION_SCALARS has 4 scalars for
        // every kind), so they are always decoded. Kinds that ignore them
        // (RELU, SiLU, ...) simply never read the result inside emunpu_activation;
        // callers that don't need them pass the FLOAT_BITS encoding of 0.0f.
        int32_t alpha_bits = args[scalars_start+2];
        int32_t beta_bits  = args[scalars_start+3];

        float alpha_f, beta_f;
        memcpy(&alpha_f, &alpha_bits, sizeof(float));   // reinterpret the bits, not convert the value
        memcpy(&beta_f,  &beta_bits,  sizeof(float));

        return runActivation(buffer_start, args, count, kind, line, double(alpha_f), double(beta_f));
    }



     Error emunpuDevice::do_vector_op(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){

         uint32_t count = args[scalars_start+0];
         emunpu_ew_kind_t kind = (emunpu_ew_kind_t)args[scalars_start+1];
         uint32_t bcast_period = args[scalars_start+2];

         bool is_binary = emunpu_ew_is_binary(kind) != 0;  // true for kinds that read both A and B, false for unary kinds that ignore B.
         bool has_b = (num_buffers == 3);

         // Check that the number of buffers matches the kind.
         if(is_binary && !has_b){
             return makeError("VECTOR_OP kind needs an operand B buffer (3 buffers), got 2", line);
         }
         if(!is_binary && has_b){
             return makeError("VECTOR_OP kind is unary and takes no operand B buffer, got 3", line);
         }

         emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
         emunpu_tensor_id_t c = buffer_table[args[buffer_start + (has_b ? 2 : 1)]].real_id;
         // 0 is emunpu's "no tensor" sentinel for a unary kind -- the engine
         // ignores b entirely when emunpu_ew_is_binary(kind) is false.
         emunpu_tensor_id_t b = has_b ? buffer_table[args[buffer_start+1]].real_id : 0;

         RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "VECTOR_OP input A", line));
         if(has_b){
             // With bcast_period != 0, the device only reads `bcast_period`
             // elements of B and repeats them (core.c: b_elems = period ? period : n).
             // Requiring B to hold `count` elements would reject a correctly
             // sized, legitimately smaller broadcast buffer.
             int64_t b_elements = bcast_period != 0 ? (int64_t)bcast_period : (int64_t)count;
             RETURN_IF_ERROR(requireElements(args[buffer_start+1], b_elements, "VECTOR_OP input B", line));
         }
         RETURN_IF_ERROR(requireElements(args[buffer_start + (has_b ? 2 : 1)], (int64_t)count, "VECTOR_OP output", line));

         emunpu_status_t status = emunpu_vector_op(emu_device, nullptr, c, a, b, count, kind, bcast_period);
         if(status != EMUNPU_OK){
             return makeError("emunpu_vector_op failed", line);
         }

         return ok();
    }


    Error emunpuDevice:: do_reduce_op(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){

emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
    emunpu_tensor_id_t c = buffer_table[args[buffer_start + 1]].real_id;

    uint32_t rows = args[scalars_start+0];
    uint32_t width = args[scalars_start+1];
    emunpu_reduce_kind_t kind = (emunpu_reduce_kind_t)args[scalars_start+2];

    RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)rows*width, "REDUCE input", line));
    RETURN_IF_ERROR(requireElements(args[buffer_start + 1], (int64_t)rows, "REDUCE output", line));

emunpu_status_t status =emunpu_reduce(emu_device, nullptr, c, a,
                                                    rows, width,kind);


if(status != EMUNPU_OK){
    return makeError("emunpu_reduce failed", line);     
}

return ok();
    }




        Error emunpuDevice:: do_matvec(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){
           emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
           emunpu_tensor_id_t x = buffer_table[args[buffer_start+1]].real_id;
           emunpu_tensor_id_t y = buffer_table[args[buffer_start+2]].real_id;
           uint32_t M = args[scalars_start+0];
           uint32_t K = args[scalars_start+1];

           RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)M*K, "MATVEC A", line));
           RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)K, "MATVEC X", line));
           RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)M, "MATVEC Y", line));    





emunpu_status_t status = emunpu_matvec(emu_device, nullptr,
                              y,  a,
                               x,  M, K);

        if(status != EMUNPU_OK){
            return makeError("emunpu_matvec failed", line); 
        }


        return ok();


        }


             //  c += a * b, in-place
Error emunpuDevice::do_fma(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){


emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
        emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
        emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;
        uint32_t count = args[scalars_start+0];

        RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)count, "FMA A", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)count, "FMA B", line));
        RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)count, "FMA C", line));



emunpu_status_t status = emunpu_fma(emu_device, nullptr,
                           c,a,b, count);


        if(status != EMUNPU_OK){
            return makeError("emunpu_fma failed", line); 
        }


        return ok();


        }
             //C = relu? relu( A*B + bias ) : ( A*B + bias )  // no relu
          Error emunpuDevice:: do_gemm(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line){

            uint32_t M = args[scalars_start+0];
            uint32_t K = args[scalars_start+1];
            uint32_t N = args[scalars_start+2];
            int transpose_b = args[scalars_start+3];
            int fuse_relu = args[scalars_start+4];




            emunpu_tensor_id_t a = buffer_table[args[buffer_start+0]].real_id;
            emunpu_tensor_id_t b = buffer_table[args[buffer_start+1]].real_id;
            emunpu_tensor_id_t c = buffer_table[args[buffer_start+2]].real_id;


            RETURN_IF_ERROR(requireElements(args[buffer_start+0], (int64_t)M*K, "GEMM A", line));
            RETURN_IF_ERROR(requireElements(args[buffer_start+1], (int64_t)K*N, "GEMM B", line));
            RETURN_IF_ERROR(requireElements(args[buffer_start+2], (int64_t)M*N, "GEMM C", line));  

            bool has_bais = (num_buffers == 4);
            emunpu_tensor_id_t bias=0; // 0 tells emunpu "no bias"

            if(has_bais){

            bias = buffer_table[args[buffer_start+3]].real_id;
            RETURN_IF_ERROR(requireElements(args[buffer_start+3], (int64_t)N, "GEMM bias", line));

            }


 




emunpu_status_t  status = emunpu_gemm(emu_device, nullptr,
                          c, a, b,  bias,M,  N,  K,  transpose_b, fuse_relu);
                if(status != EMUNPU_OK){
                    return makeError("emunpu_gemm failed", line); 
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





