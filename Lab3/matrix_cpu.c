#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
 
void matrixMultiplyCPU(float *A, float *B, float *C, int N) { 
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) { 
            float sum = 0.0f; 
            for (int k = 0; k < N; k++) { 
                sum += A[i * N + k] * B[k * N + j]; 
            } 
            C[i * N + j] = sum; 
        } 
    } 
} 
 
int main(int argc, char **argv) { 
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input 
    size_t size = N * N * sizeof(float); 
 
    // Use heap instead of stack (int A[N][N]) for large matrices
    // If use stack, may cause stack overflow for large N
    // and A[N][N] is not standard C, may not compile
    float *A = (float *)malloc(size); // allocate memory with given size and return the pointer
    float *B = (float *)malloc(size); 
    float *C = (float *)malloc(size); 
 
    for (int i = 0; i < N * N; i++) { 
        A[i] = rand() % 100 / 100.0f; 
        B[i] = rand() % 100 / 100.0f; 
    } 
 
    clock_t start = clock(); 
    matrixMultiplyCPU(A, B, C, N); 
    clock_t end = clock(); 
 
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC; 
    printf("%d %f\n", N, elapsed);

    // Volatile to prevent optimization (gvc -02) removing the loop
    volatile float sink = 0.0f;
    for (int i = 0; i < N * N; i++) {
        sink += C[i];
    }
 
    free(A); free(B); free(C); // free the allocated memory
    return 0; 
}
