%%writefile vector_mm.cu
#include<stdio.h>
#include "timer.h"
#include <cuda_runtime.h>

// Executes on the CPU
void mm_CPU(float* x, float* y, float* z, int n) {
    for(int row=0;row<n;row++){
        for(int col=0;col<n;col++){
            float sum=0;
            for(int i=0;i<n;i++){
                sum+=x[row*n + i]*y[i*n + col];
            }
            z[row*n + col] = sum;
        }
    }
} 

// Kernel: Executes on the GPU
__global__ void vec_mm_Kernel(float* x, float* y, float* z, int n,int width,int height) {

    int col = blockDim.x * blockIdx.x + threadIdx.x; // finding the global thread index
    int row = blockDim.y*blockIdx.y + threadIdx.y;



  if(row<height && col<width){

    int out_indx =row*width + col;

    float sum=0;
    for(int i=0;i<n;i++){
  sum+=x[row*width + i]*y[i*width + col];
    }
    z[out_indx] = sum;
}

}

// Host: Executed on the CPU
void mm_GPU(float* x, float* y, float* z,int height,int width, int n) {
    float *x_d, *y_d, *z_d;
    // Allocate device memory
    size_t size = n * n * sizeof(float);

    cudaMalloc(&x_d, size);
    cudaMalloc(&y_d, size);
    cudaMalloc(&z_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    // 3. Launch Kernel
    StartTimer();
   dim3 blockDim(4,4); // 4 threads in x and y direction
   dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1 )/ blockDim.y);
    
    vec_mm_Kernel<<<gridDim, blockDim>>>(x_d, y_d, z_d, n,width, height);
    cudaDeviceSynchronize();
    StopTimer();

    printf("only  kernel computation time on. gpu: %f s\n", GetElapsedSeconds());

    

    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main() {
    int n = 1024;
    size_t size = n*n* sizeof(float);

    float* x_h = (float*)malloc(size);
    float* y_h = (float*)malloc(size);
    float* z_h_gpu = (float*)malloc(size);
    float* z_h_cpu = (float*)malloc(size);

    for (int i = 0; i < n*n; i++) {
        x_h[i] = 1.0f * i;
        y_h[i] = 2.0f * i;
    }
    int height = n;
    int width = n;
    // Perform CPU vector addition
    printf("--- Running CPU Vector Add ---\n");
    StartTimer();
    mm_CPU(x_h, y_h, z_h_cpu, n);
    StopTimer();
    printf("CPU VecAdd Time: %f s\n", GetElapsedSeconds());

    // Perform GPU vector addition
    printf("\n--- Running GPU Vector Add ---\n");
    StartTimer();
    mm_GPU(x_h, y_h, z_h_gpu,n,n, n);
    StopTimer();
    printf("Total GPU VecAdd Time (including memory transfers): %f s\n", GetElapsedSeconds());

    // Free host memory
    free(x_h);
    free(y_h);
    free(z_h_gpu);
    free(z_h_cpu);

    return 0;
}

// !nvcc -o vector_add vector_add.cu
// !./vector_add
