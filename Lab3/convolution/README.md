# Convolution (CPU & CUDA Image Processing Pipeline)

This project implements 2D image convolution using both CPU (C) and GPU (CUDA).
It supports standalone CUDA execution as well as a CUDA shared library accessed
from Python for flexible experimentation and performance comparison.

---

## Directory Structure

convolution/
├── Image_original/         Original reference images
├── Image_raw/              Raw image data used for convolution
├── Image_result/           Output images after convolution
├── convolution.c           CPU implementation (C)
├── convolution.cu          CUDA implementation (standalone)
├── convolution_cuda_lib.cu CUDA implementation compiled as shared library
├── libconv_cuda.so         Compiled CUDA shared library
├── convolution_cuda.py     Python wrapper for CUDA shared library
├── image_processing.py     Image preprocessing and postprocessing
├── visualize_raw_file.py   Raw image visualization utility
├── run_c.sh                Run CPU convolution pipeline
├── run_cuda.sh             Run CUDA convolution pipeline
├── run_cuda_lib.sh         Run CUDA via shared library
└── __pycache__/            Python cache files

---

## Requirements

### System
- Linux (native or cloud VM)
- NVIDIA GPU
- CUDA Toolkit (11.x or newer)
- gcc / nvcc

### Python Packages
pip install numpy pillow

---

## Build Instructions

### Compile CPU Version
gcc convolution.c -o convolution_c -O2

### Compile CUDA Standalone Version
nvcc convolution.cu -o convolution_cuda

### Compile CUDA Shared Library
nvcc -Xcompiler -fPIC -shared convolution_cuda_lib.cu -o libconv_cuda.so

---

## Run Instructions

### CPU Pipeline
./run_c.sh

### CUDA Standalone Pipeline
./run_cuda.sh

### CUDA Shared Library (Python)
./run_cuda_lib.sh

### Specify Kernel Type (Optional)
./run_c.sh edge
./run_cuda.sh blur
./run_cuda.sh sharpen

---

## Processing Pipeline

1. Load images from Image_original/
2. Convert images to raw format → Image_raw/
3. Apply convolution (CPU or GPU)
4. Save processed images → Image_result/
5. Optional visualization using visualize_raw_file.py

---

## Notes

- GPU execution has fixed overhead (memory copy + kernel launch)
- Small images may run slower on GPU than CPU
- Performance improves as image size increases

---

## Debugging

Check GPU and CUDA setup:
nvidia-smi
nvcc --version

Check shared library linkage:
ldd libconv_cuda.so

If Python cannot find the shared library:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

---

## Project Goals

- Compare CPU vs GPU performance
- Practice CUDA kernel programming and memory management
- Integrate CUDA with Python using shared libraries

---

## Author

Howard-Pai
