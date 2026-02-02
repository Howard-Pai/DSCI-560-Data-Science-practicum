#!/bin/bash

# Compile CPU code
gcc matrix_cpu.c -O2 -o matrix_cpu

OUTFILE="runtime_cpu.txt"
echo "# N runtime_seconds" > $OUTFILE

for N in 128 256 512 1024 2048
do
    ./matrix_cpu $N >> $OUTFILE
done

echo "CPU benchmarks done."
