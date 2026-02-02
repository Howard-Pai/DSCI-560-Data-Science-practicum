#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 

void convolution2D(unsigned int *image, unsigned int *output, int *kernel, int M, int N, int divisor) {
    int pad = N / 2;  // padding for 'same' output size

    // Outer loop: iterate over each pixel in the image 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int sum = 0;

            // Inner loop: apply the kernel
            for (int ki = 0; ki < N; ki++) {
                for (int kj = 0; kj < N; kj++) {
                    // The convolution kernel is centered at each output pixel.
                    int x = i + ki - pad;
                    int y = j + kj - pad;

                    // For boundary pixels, parts of the kernel fall outside the image 
                    // and are implicitly treated as zero-padding.
                    if (x >= 0 && x < M && y >= 0 && y < M) {
                        sum += image[x * M + y] * kernel[ki * N + kj];
                    }
                }
            }
            
            // Normalize by divisor (important for blur kernels)
            if (divisor != 1) {
                sum = sum / divisor;
            }
            
            // Clamp the output to prevent overflow
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[i * M + j] = (unsigned int)sum;
        }
    }
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

    clock_t start = clock();
    convolution2D(image, output, kernel, M, N, divisor);
    clock_t end = clock();
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    printf("Convolution time: %f ms\n", time_ms);

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

    free(image);
    free(output);
    return 0;
}