#!/bin/bash

# Compile CUDA code
nvcc matrix_cuBLAS.cu -O2 -o matrix_cuBLAS

OUTFILE="runtime_cuBLAS.txt"
echo "# N runtime_seconds" > $OUTFILE

for N in 512 1024 2048 4096 8192
do
    ./matrix_cuBLAS $N >> $OUTFILE
done

echo "cuBLAS benchmarks done."