import sys
import ctypes
import numpy as np
import time
import os

if len(sys.argv) != 4:
    print("Usage: python convolution_cuda.py <input.raw> <output.raw> <kernel>")
    sys.exit(1)

input_raw = sys.argv[1]
output_raw = sys.argv[2]
kernel = sys.argv[3].encode("utf-8")

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libconv_cuda.so")

lib.run_convolution_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_int,
    ctypes.c_char_p
]

M = 800

image = np.fromfile(input_raw, dtype=np.uint32)
output = np.zeros_like(image)

start = time.time()
lib.run_convolution_cuda(
    image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
    output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
    M,
    kernel
)
end = time.time()

print(f"Python + CUDA library time: {(end-start)*1000:.3f} ms")

output.tofile(output_raw)
