#include<iostream>

#include "iree/runtime/api.h"

using namespace std;

///explaining the model input/output shapes
 struct TensorShape{
        iree_hal_dim_t dims[4];
        iree_host_size_t rank;
    };

    struct Modelsignature{
      const char* function_name;
      TensorShape inputs[8];
      iree_host_size_t num_inputs;
      TensorShape output;
    };


//forward declarations

static int iree_runtime_main(void);
static iree_status_t iree_runtime_run_session(
    iree_runtime_instance_t* instance);

static iree_status_t make_input_buffer_view(
    iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator,
    const float* data,
    const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank,
    iree_host_size_t data_size_bytes,
    iree_hal_buffer_view_t** out_buffer_view);

static iree_status_t run_model(
    iree_runtime_session_t* session, const Modelsignature* sig,
    const float* const* inputs, float* C);



   


const char* file_path = "";



iree_host_size_t size_from_shape (const TensorShape* shape) {

  iree_host_size_t size = 1;
  for (iree_host_size_t i = 0; i < shape->rank; ++i) {
    size *= shape->dims[i];
  }
  return size*sizeof(float);
}


//loading module from  file
 iree_status_t iree_runtime_load_module(
    iree_runtime_session_t* session) {
  return iree_runtime_session_append_bytecode_module_from_file(session,
                                                               file_path);
}


//////////// main

int main(int argc ,char** argv){

if(argc < 2){
    cout<<"provide the module file path\n";
    return 1;
}

file_path = argv[1];

    return iree_runtime_main();
}


///////1.  iree_runtime_instance_t setup
int iree_runtime_main(){

    iree_runtime_instance_options_t instance_options;     //Options used to configure instance creation.
    iree_runtime_instance_options_initialize(&instance_options);

    iree_runtime_instance_options_use_all_available_drivers(&instance_options);  //Register every driver that was compiled into this runtime.


    iree_runtime_instance_t* instance = NULL;   // An instance is the top-level runtime object.
    iree_status_t status = iree_runtime_instance_create(          //The function allocates the instance on the heap and stores its address in instance.
        &instance_options, iree_allocator_system(), &instance);  


    if (iree_status_is_ok(status)) {    //Every IREE API returns a status.
      status = iree_runtime_run_session(instance);    // now creating a session, loading the model, and running it.(passing the instance to the session)
    }


    // Release the shared instance - it will be deallocated when all sessions
    // using it have been released (here it is deallocated immediately).
    iree_runtime_instance_release(instance);


    int ret = (int)iree_status_code(status);

    if (!iree_status_is_ok(status)) {
      // Dump nice status messages to stderr on failure.
      // An application can route these through its own logging infrastructure as
      // needed. Note that the status is a handle and must be freed!
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
    }
    return ret;

}

///2. load module


iree_status_t iree_runtime_run_session(
    iree_runtime_instance_t* instance) {

//creating device (here CPU) that will actually execute our program.
        iree_hal_device_t* device = NULL;
        IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
            instance, iree_make_cstring_view("local-task"), &device));



        iree_runtime_session_options_t session_options;
        iree_runtime_session_options_initialize(&session_options);

        iree_runtime_session_t* session = NULL;
        iree_status_t status = iree_runtime_session_create_with_device(//Creating session with the device
            instance, &session_options, device,
            iree_runtime_instance_host_allocator(instance), &session);  //IREE must allocate the session object itself and its internal host-side data structures,
        
        iree_hal_device_release(device);



        if (iree_status_is_ok(status)) {
          status = iree_runtime_load_module(session);  //loading module to the session 
        }

// preparing data for the call to the function in the module.


      // int M=2;
      // int K=2;
      // int N=2;
      const float A[10] = {5.0f, -3.0f, 3.0f, 4.0f};
      //const float B[4] = {1.0f, 2.0f, 3.0f, -6.0f};
      float C[5];

      Modelsignature sig = {
        .function_name = "module.mlp_forward",
        .inputs = {
            {.dims = {1, 10}, .rank = 2},
            //{.dims = {2, 2}, .rank = 2}
        },
        .output = {.dims = {1, 5}, .rank = 2},
        .num_inputs=1,
      };

        // Build and issue the call.
        if (iree_status_is_ok(status)) {
          const float* const inputs[] = {A};
          status = run_model(session, &sig, inputs, C);
        }
      
        // Release the session and free all resources.
        iree_runtime_session_release(session);
        return status;

}

// making input tensors  on device memory and copying the data from host to device memory.
// The function returns a buffer view that describes the tensor in device memory.
static iree_status_t make_input_buffer_view( iree_hal_device_t* device, iree_hal_allocator_t* device_allocator, const float* data,
    const iree_hal_dim_t* shape,  iree_host_size_t shape_rank, 
    iree_host_size_t data_size_bytes,
    iree_hal_buffer_view_t** out_buffer_view){

    
  iree_status_t status = iree_ok_status();


    if (iree_status_is_ok(status)) {

   
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          shape_rank, shape, // tells the function how many dimensions and what they are.
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){   //compound literal in C.(Allocate a normal tensor buffer in device memory with unrestricted access.)
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory:
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
         
          iree_make_const_byte_span(data, data_size_bytes),   // It describes the source data to copy.
          // Buffer view + storage are returned and owned by the caller:
          out_buffer_view);
    }



  return status;
}



/////////// 3. Call a function within a module with buffer views
static iree_status_t run_model( iree_runtime_session_t* session, const Modelsignature* sig, const float* const* inputs,float* C) {

  // Initialize the call to the function.  
  iree_runtime_call_t call;              
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(sig->function_name), &call));


      //debugging statement to print the function signature of the function we are calling.
  iree_string_view_t abi_decl = iree_vm_function_lookup_attr_by_name(
      &call.function, iree_make_cstring_view("iree.abi.declaration"));
  cout << "iree.abi.declaration = '"
       << string(abi_decl.data, abi_decl.size) << "'" << endl;

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status = iree_ok_status();


  for(int i=0;i<sig->num_inputs;i++){
  iree_hal_buffer_view_t* buffInput = NULL;

   if (iree_status_is_ok(status))
 {
  
   status = make_input_buffer_view(device, device_allocator, inputs[i], sig->inputs[i].dims, sig->inputs[i].rank, size_from_shape(&sig->inputs[i]), &buffInput);
iree_runtime_call_inputs_push_back_buffer_view(&call, buffInput); //attach the buffer view(input tensor) to the call inputs.

 iree_hal_buffer_view_release(buffInput);

 }


  }





// iree_hal_dim_t shape_A[2] = {(iree_hal_dim_t)M, (iree_hal_dim_t)K};

// iree_hal_dim_t shape_B[2] = {(iree_hal_dim_t)K, (iree_hal_dim_t)N};

//  if (iree_status_is_ok(status))
//  {
  
//    status = make_input_buffer_view(device, device_allocator, A, sig->inputs[0].dims, sig->inputs[0].rank, size_from_shape(&sig->inputs[0]), &bufA);
// iree_runtime_call_inputs_push_back_buffer_view(&call, bufA); //attach the buffer view(input tensor) to the call inputs.

//  iree_hal_buffer_view_release(bufA);

//  }




//  if(status == iree_ok_status())
//  {
  
//    status = make_input_buffer_view(device, device_allocator, B, sig->inputs[1].dims, sig->inputs[1].rank,size_from_shape(&sig->inputs[1]), &bufB);
//    iree_runtime_call_inputs_push_back_buffer_view(&call, bufB);

//  iree_hal_buffer_view_release(bufB);

//  }



  // perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0); // here computation is done on the device and the result is stored in device memory.
  }



  // fprintf(stdout, "\n = \n");


  // Dump the function outputs.
  iree_hal_buffer_view_t* ret0 = NULL;
  if (iree_status_is_ok(status)) {
    // try to get the call result as a buffer view. from call object to ret0 
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0);
  }
// Copy the result back to the host in C array
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(ret0), 0,
     C, size_from_shape(&sig->output), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    status = iree_hal_buffer_view_fprint(
        stdout, ret0, /*max_element_count=*/4096, host_allocator);
  }
  iree_hal_buffer_view_release(ret0);

  iree_runtime_call_deinitialize(&call);
  return status;
}
