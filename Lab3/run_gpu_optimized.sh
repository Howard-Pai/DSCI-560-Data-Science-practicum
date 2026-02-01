#!/bin/bash

# Compile CUDA code
nvcc matrix_gpu_optimized.cu -O2 -o matrix_gpu

OUTFILE="runtime_gpu_optimized.txt"
echo "# N runtime_seconds" > $OUTFILE

for N in 128 256 512 1024 2048
do
    ./matrix_gpu $N >> $OUTFILE
done

echo "GPU benchmarks done."