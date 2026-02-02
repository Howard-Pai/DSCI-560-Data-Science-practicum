#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__constant__ int d_kernel[9];

// same as in convolution.cu
__global__ void convolution2D_GPU(
    unsigned int *image,
    unsigned int *output,
    int M,
    int N,
    int divisor
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= M) return;

    int pad = N / 2;
    int sum = 0;

    for (int ki = 0; ki < N; ki++) {
        for (int kj = 0; kj < N; kj++) {
            int x = i + ki - pad;
            int y = j + kj - pad;

            if (x >= 0 && x < M && y >= 0 && y < M) {
                sum += image[x * M + y] * d_kernel[ki * N + kj];
            }
        }
    }

    if (divisor != 1)
        sum /= divisor;

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;

    output[i * M + j] = (unsigned int)sum;
}

extern "C" void run_convolution_cuda(
    unsigned int *h_image,
    unsigned int *h_output,
    int M,
    const char *kernel_type
) {
    int kernel[9];
    int divisor = 1;
    int N = 3;

    if (strcmp(kernel_type, "blur") == 0) {
        int k[9] = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        memcpy(kernel, k, sizeof(k));
        divisor = 9;
    } else if (strcmp(kernel_type, "sharpen") == 0) {
        int k[9] = {
             0,-1, 0,
            -1, 5,-1,
             0,-1, 0
        };
        memcpy(kernel, k, sizeof(k));
    } else if (strcmp(kernel_type, "emboss") == 0) {
        int k[9] = {
            -2,-1, 0,
            -1, 1, 1,
             0, 1, 2
        };
        memcpy(kernel, k, sizeof(k));
    } else {
        int k[9] = {
            -1,-1,-1,
            -1, 8,-1,
            -1,-1,-1
        };
        memcpy(kernel, k, sizeof(k));
    }

    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(kernel));

    unsigned int *d_image, *d_output;
    cudaMalloc(&d_image, M*M*sizeof(unsigned int));
    cudaMalloc(&d_output, M*M*sizeof(unsigned int));

    cudaMemcpy(d_image, h_image, M*M*sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((M+15)/16, (M+15)/16);

    convolution2D_GPU<<<blocks, threads>>>(d_image, d_output, M, N, divisor);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, M*M*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);
}
