#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // Added for strcmp and memcpy
#include <cuda_runtime.h>

// Declare constant memory for kernel (faster access)
__constant__ int d_kernel[9];

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
                sum += image[x * M + y] * d_kernel[ki * N + kj];  // Use d_kernel from constant memory
            }
        }
    }

    if (divisor != 1)
        sum /= divisor;

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;

    output[i * M + j] = (unsigned int)sum;
}


int main(int argc, char *argv[]) {

    int M = 800;  // Image size
    int N = 3;    // Kernel size

    unsigned int *image = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        perror("Failed to open input image");
        return -1;
    }

    // Fread: read the image data
    fread(image, sizeof(unsigned int), M * M, fp);
    fclose(fp);

    unsigned int *output = (unsigned int *)malloc(M * M * sizeof(unsigned int));

    // Default kernel type is "edge"
    char *kernel_type = "edge";
    if (argc >= 4) {
        kernel_type = argv[3];
    }

    // Select kernel based on type
    int kernel[9];
    int divisor = 1;  // Normalization divisor
    
    if (strcmp(kernel_type, "blur") == 0) {
        // Blur kernel - needs division by 9 to average the pixels
        int blur_kernel[9] = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        memcpy(kernel, blur_kernel, sizeof(blur_kernel));
        divisor = 9;  // Sum of all kernel values
        printf("Using blur kernel (divisor: %d)\n", divisor);
    }
    else if (strcmp(kernel_type, "sharpen") == 0) {
        // Sharpen kernel
        int sharpen_kernel[9] = {
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        };
        memcpy(kernel, sharpen_kernel, sizeof(sharpen_kernel));
        divisor = 1;
        printf("Using sharpen kernel\n");
    }
    else if (strcmp(kernel_type, "emboss") == 0) {
        // Emboss kernel
        int emboss_kernel[9] = {
            -2, -1,  0,
            -1,  1,  1,
             0,  1,  2
        };
        memcpy(kernel, emboss_kernel, sizeof(emboss_kernel));
        divisor = 1;
        printf("Using emboss kernel\n");
    }
    else {
        // Edge detection kernel (default)
        int edge_kernel[9] = {
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        };
        memcpy(kernel, edge_kernel, sizeof(edge_kernel));
        divisor = 1;
        printf("Using edge detection kernel (default)\n");
    }

    cudaMemcpyToSymbol(d_kernel, kernel, N * N * sizeof(int));

    // Allocate device memory
    unsigned int *d_image, *d_output;
    cudaMalloc((void**)&d_image, M * M * sizeof(unsigned int));
    cudaMalloc((void**)&d_output, M * M * sizeof(unsigned int));

    // Copy image to device
    cudaMemcpy(d_image, image, M * M * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Measure GPU runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution2D_GPU<<<numBlocks, threadsPerBlock>>>(
        d_image, d_output, M, N, divisor
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("GPU Convolution time: %f ms\n", time_ms);

    // Copy result back to host
    cudaMemcpy(output, d_output, M * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        perror("Failed to open output file");
        free(image);
        free(output);
        return -1;
    }
    fwrite(output, sizeof(unsigned int), M * M, out);
    fclose(out);

    printf("Output saved to %s\n", argv[2]);

    cudaFree(d_image);
    cudaFree(d_output);
    free(image);
    free(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}