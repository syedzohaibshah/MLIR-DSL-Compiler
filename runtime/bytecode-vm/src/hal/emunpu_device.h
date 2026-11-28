#pragma once

#include "device.h"
#include "emunpu/device.h"   // external library header needed to specify path at compile time
#include <vector>
#include <string>

struct TensorSlot {
    emunpu_tensor_id_t real_id;
    bool freed;
    int64_t bytes;   // size given to ALLOC; kernel shape checks read it
};

// What one LAUNCH scalar means.
enum class ScalarKind {
    DIM,    // a count or a dimension; must be > 0
    FLAG,   // a mode selector; 0 or 1 only
    PARAM,  // a tuning value; any int, negative allowed
    ENUM,   // selects one of a fixed set of kinds; 0 .. enum_count-1
    FLOAT_BITS,  // a float value
   
};

// One scalar position's kind, plus how many values are valid when the kind
// is ENUM (ignored otherwise). Kept here, next to ScalarKind, because the
// valid range for an ENUM differs per kernel (e.g. 20 activation kinds vs
// 9 elementwise kinds) and has to travel with the position, not the kind.
struct ScalarSpec {
    ScalarKind kind;
    int enum_count;   // only meaningful when kind == ScalarKind::ENUM
};

// One row per op_id. dispatch() validates against this row, so each do_* body
// reads arguments that are already known to be the right count and range.
// A kernel with an optional buffer gives a min and max that differ.
struct KernelSignature {
    const char* name;
    int min_buffers;
    int max_buffers;
    const ScalarSpec* scalars;   // one entry per scalar position, in order
    int num_scalars;
};

// Returns nullptr when op_id has no row (unknown kernel).
const KernelSignature* lookupKernelSignature(int op_id);

class emunpuDevice:public Device{

    emunpu_device_t* emu_device;
    std::vector<TensorSlot> buffer_table;
    Error isValidHandle(DeviceHandle h, int line);

    // Checks that buffer `h` holds at least `elements` fp32 values.
    Error requireElements(DeviceHandle h, int64_t elements, const char* what, int line);


    // Kernels. dispatch() checks the buffer and scalar counts against the
    // signature table before it calls one, so no kernel repeats that check.
    // Only RMS_NORM reads num_buffers, to tell its 2- and 3-buffer forms apart.

    Error do_matmul(int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    Error do_vec_add(int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    Error do_bais_add(int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    Error do_rms_norm(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    Error do_softmax(int scalars_start,int buffer_start,const std::vector<int>& args,int line);
    Error do_transpose(int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    // Shared by RELU (op_id 2) and ACTIVATION (op_id 7): both end in the same
    // emunpu_activation call, differing only in which kind is fixed vs chosen
    // by a LAUNCH scalar. buffer_start indexes into args for the two buffers
    // (input, output); count/kind are passed directly, not read from args,
    // since RELU's LAUNCH has no kind scalar at all.
    Error runActivation(int buffer_start,const std::vector<int>& args,
                        uint32_t count,emunpu_act_kind_t kind,int line,double alpha=0.0,double beta=0.0);

    /*.   do_relu just call the do activation with the kind fixed to RELU, 
    do activation can also handle it so it is duplication , but it is not removed because it will break the order 
    of the kernel table and the op_id mapping, so it is kept for now. 
    */
    Error do_relu(int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    // General activation kernel. Reads kind (and alpha/beta, always present as
    // FLOAT_BITS scalars) from args and calls runActivation. Supports all 24
    // emunpu_act_kind_t kinds.
    Error do_activation(int scalars_start,int buffer_start,const std::vector<int>& args,int line);


    // Elementwise sub-operation (EMUNPU_I_EW). num_buffers tells apart a unary
    // kind's LAUNCH (2 buffers: a, c) from a binary kind's (3: a, b, c), the
    // same way RMS_NORM's num_buffers tells its 2- and 3-buffer forms apart.
    Error do_vector_op(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);

    Error do_reduce_op(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);
        //  c=A*v   (A -> matrix , v -> vector)
        Error do_matvec(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);

      //  c += a * b, in-place
        Error do_fma(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);

        //C = relu?( A*B + bias ) 
        Error do_gemm(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);

        // Layer normalization over the last dimension:
//   c = (a - mean(a)) / sqrt(var(a) + eps) * gamma + beta
// |gamma| / |beta| are length-|width| vectors broadcast across rows; either may
// be 0 to skip that term.
        Error do_layer_norm(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);


        // A full linear layer: C = act?( A*B (+ bias) ), i.e. emunpu_gemm generalized
// from ReLU-only to any activation. |bias| may be 0 and |act| may be NULL.
        Error do_linear(int num_buffers,int scalars_start,int buffer_start,const std::vector<int>& args,int line);


   
public:
    emunpuDevice();
    ~emunpuDevice() override;
 Error status() const;
    
    std::string name() const override;
    DeviceLimits limits() const override;
    
     Error allocate(int size,int line,DeviceHandle& out_handle) override;

     Error free(DeviceHandle handle,int line)override;
     Error upload(DeviceHandle handle, const void* src, int size, int line)override;
    Error download(DeviceHandle handle, void* dst, int size, int line) override;
    Error dispatch(const std::vector<int>& args, int line) override;

};