import matplotlib.pyplot as plt

def load(filename):
    N, T = [], []
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            n, t = line.split()
            N.append(int(n))
            T.append(float(t))
    return N, T

N_cpu, T_cpu = load("runtime_cpu.txt")
N_gpu, T_gpu = load("runtime_gpu.txt")
N_optimized, T_optimized = load("runtime_gpu_optimized.txt")
N_cuBLAS, T_cuBLAS = load("runtime_cuBLAS.txt")

plt.figure()
plt.plot(N_cpu, T_cpu, marker='o', label="CPU")
plt.plot(N_gpu, T_gpu, marker='s', label="GPU")
plt.plot(N_optimized, T_optimized, marker='^', label="GPU Optimized")
plt.plot(N_cuBLAS, T_cuBLAS, marker='d', label="cuBLAS")

plt.xlabel("Matrix size N")
plt.ylabel("Execution time (seconds)")
plt.title("CPU vs GPU vs GPU Optimized Matrix Multiplication Runtime")
plt.yscale("log") 
plt.legend()
plt.grid(True)

plt.savefig("runtime_comparison.png")
plt.show()
