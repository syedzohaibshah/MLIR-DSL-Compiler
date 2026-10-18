#include "kernels.h"

// z[row][col] = sum_i x[row][i] * y[i][col]
__global__ void vec_mm_Kernel(float* x, float* y, float* z, int n, int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row < height && col < width){
        int out_indx = row * width + col;
        float sum = 0;
        for(int i = 0; i < n; i++){
            sum += x[row * width + i] * y[i * width + col];
        }
        z[out_indx] = sum;
    }
}

// z[i] = x[i] + y[i], one thread per element.
__global__ void vec_add_Kernel(float* x, float* y, float* z, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        z[i] = x[i] + y[i];
    }
}

// z[i] = max(x[i], 0), one thread per element. Standard activation after a
// linear layer's matmul + bias add.
__global__ void relu_Kernel(float* x, float* z, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        z[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// z[row][col] = x[row][col] + bias[col], one thread per element.
// The other half of a linear layer: matmul produces x, this adds the
// per-output-feature bias before the activation runs.
__global__ void bias_add_Kernel(float* x, float* bias, float* z, int n, int out_features) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        int col = i % out_features;
        z[i] = x[i] + bias[col];
    }
}
