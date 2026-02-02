#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// global: function to be executed on the GPU
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) { 
    // block index x/y : the index of the block in the grid along x/y dimension
    // thread index x/y: the index of the thread in the block along x/y dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    // calculate C[row][col]
    if (row < N && col < N) { 
        float sum = 0.0f; 
        for (int k = 0; k < N; k++) { 
            sum += A[row * N + k] * B[k * N + col]; 
        } 
        C[row * N + col] = sum; 
    } 
} 

int main(int argc, char **argv){
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = N * N * sizeof(float); 

    float *A, *B, *C;
    cudaMallocManaged(&A, size); // CUDA cannot access host memory allocated by malloc, so use cudaMallocManaged
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N * N; i++) { 
        A[i] = rand() % 100 / 100.0f; 
        B[i] = rand() % 100 / 100.0f; 
    } 

    // Define block and grid sizes
    // CUDA has grid -> block -> thread(worker) hierarchy
    // Each thread is respnsible for calculating one element in C
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // GPU timing using cudaEvent_t
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyGPU<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%d %f\n", N, milliseconds/ 1000.0f);

    volatile float sink = 0.0f;
    for (int i = 0; i < N * N; i++) {
        sink += C[i];
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
