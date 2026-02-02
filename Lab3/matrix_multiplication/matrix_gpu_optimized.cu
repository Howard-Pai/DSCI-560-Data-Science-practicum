#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16 // meaning each block handles 16 * 16 threads

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // shared memory = memory on GPU
    //      - faster than global memory
    //      - shared among threads in the same block
    // ds_a/ ds_b: sub-matrix (tile) of A and B.
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { // m: index of the tile
        // Assign data into shared memory, each thread only load one element
        // The if condition is just to make sure we do not access out of bound memory
        if (Row < N && (m*TILE_WIDTH+tx) < N){
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        }else{
            ds_A[ty][tx] = 0.0f; 
        }

        if (Col < N && (m*TILE_WIDTH+ty) < N){
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
        }else{
            ds_B[ty][tx] = 0.0f;
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads();

        // Multiply the tiles together
        // ex. Pvalue += A[0][0]*B[0][0] + A[0][1]*B[0][1] ... m == 0
        //     Pvalue += A[0][2]*B[2][0] + A[0][3]*B[3][0] ... m == 1 ...
        for (int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write the summed value to C
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

int main (int argc, char **argv){
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
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // GPU timing using cudaEvent_t
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyTiled<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
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