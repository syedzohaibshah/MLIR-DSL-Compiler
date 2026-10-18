#pragma once
#include <cuda_runtime.h>

// __global__ kernels, defined in kernels.cu
__global__ void vec_mm_Kernel(float* x, float* y, float* z, int n, int width, int height);
__global__ void vec_add_Kernel(float* x, float* y, float* z, int n);
__global__ void relu_Kernel(float* x, float* z, int n);
__global__ void bias_add_Kernel(float* x, float* bias, float* z, int n, int out_features);
