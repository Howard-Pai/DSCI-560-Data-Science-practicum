# Matrix Multiplication (CPU & GPU)

## Overview

This project implements **matrix multiplication** using multiple approaches to demonstrate CPU vs GPU acceleration:

1. **CPU Implementation** (`matrix_cpu.c`) – standard nested-loop multiplication.
2. **GPU Implementation** (`matrix_gpu.cu`) – basic CUDA kernel without shared memory.
3. **Optimized GPU Implementation** (`matrix_gpu_optimized.cu`) – CUDA kernel using **shared memory** for higher efficiency.
4. **cuBLAS Implementation** (`matrix_cuBLAS.cu`) – GPU implementation leveraging NVIDIA’s **cuBLAS library** for maximum performance.

The project also includes **automation scripts** to run each version and measure runtime.

---

## Directory Structure

matrix_multiplication/
├─ matrix_cpu.c
├─ matrix_gpu.cu
├─ matrix_gpu_optimized.cu
├─ matrix_cuBLAS.cu
├─ run_cpu.sh
├─ run_gpu.sh
├─ run_gpu_optimized.sh
├─ run_cuBLAS.sh
...
└─ runtime_comparison.png

## Usage

### 4 versions of matrix comparison

CPU/ GPU/ GPU with shared memory/ cuBLAS
use the corresponding .sh file to run the code.

## Performance Observations
### Summary Table

| Implementation        | Small N (≤256)                 | Large N (≥512)                  |
|-----------------------|--------------------------------|--------------------------------|
| CPU                   | Slow, O(N³)                   | Very slow                     |
| GPU Naive             | Fast, limited overhead        | Outperformed by optimized GPU |
| GPU Optimized (Tiling)| Slightly slower than naive     | Faster than naive GPU         |
| cuBLAS                | Sometimes slower (startup)    | Up to 7× faster than hand-written kernels |

**Key Insights:**
- Tiling optimization benefits mainly **large matrices**.  
- cuBLAS is faster due to highly tuned kernels, memory optimization, vectorization, auto-tuning, and overlapping compute & memory transfer.  
- Small matrix performance may be limited by **kernel launch overhead and memory alignment**.  


## Requirements

NVIDIA GPU with CUDA support

NVIDIA CUDA Toolkit (nvcc)

C compiler (gcc) for CPU version

Bash shell to run scripts (run_*.sh)

