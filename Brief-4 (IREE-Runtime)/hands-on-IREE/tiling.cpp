%%writefile vector_mm_tile.cu
#include<stdio.h>
#include "timer.h"
#include <cuda_runtime.h>

#define TILE_DIM 32

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


//shared memory for tiles
__shared__ float tile_x[TILE_DIM][TILE_DIM];
__shared__ float tile_y[TILE_DIM][TILE_DIM];

 float sum=0.f;

for(int tile=0;tile<n/TILE_DIM; tile++){

    //copying data to shared memory
tile_x[threadIdx.y][threadIdx.x]=x[row*n+tile*TILE_DIM + threadIdx.x];
tile_y[threadIdx.y][threadIdx.x]=y[(tile*TILE_DIM + threadIdx.y)*n + col];

__syncthreads();

for(int i=0;i<TILE_DIM;i++){

    sum+=tile_x[threadIdx.y][i]*tile_y[i][threadIdx.x];
}
__syncthreads();

}

z[row*width + col] = sum;

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
   dim3 blockDim(32,32); // 4 threads in x and y direction
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
    int n = 256;
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
    // Perform CPU vector MM
    printf("--- Running CPU Vector MM ---\n");
    StartTimer();
    mm_CPU(x_h, y_h, z_h_cpu, n);
    StopTimer();
    printf("CPU VecMM Time: %f s\n", GetElapsedSeconds());

    // Perform GPU vector addition
    printf("\n--- Running GPU Vector MM with tiling  ---\n");
    StartTimer();
    mm_GPU(x_h, y_h, z_h_gpu,n,n, n);
    StopTimer();
    printf("Total GPU VecMM Time (including memory transfers): %f s\n", GetElapsedSeconds());


    // Free host memory
    free(x_h);
    free(y_h);
    free(z_h_gpu);
    free(z_h_cpu);

    return 0;
}

// !nvcc -o vector_add vector_add.cu
// !./vector_add
