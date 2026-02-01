import ctypes 
import numpy as np 
import time 
 
# Load shared library 
lib = ctypes.cdll.LoadLibrary("./libmatrix.so") 
 
# Define argument types
lib.gpu_matrix_multiply.argtypes = [ 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
    ctypes.c_int 
] 
 
result = []
result.append("# N runtime_seconds\n")
for N in [128, 256, 512, 1024, 2048]:
    A = np.random.rand(N, N).astype(np.float32) 
    B = np.random.rand(N, N).astype(np.float32) 
    C = np.zeros((N, N), dtype=np.float32) 
 
    start = time.time() 
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N) 
    end = time.time() 
 
    result.append(f"{N} {end - start}\n")

with open("runtime_python_cudaLibrary.txt", "w") as f:
    f.writelines(result)