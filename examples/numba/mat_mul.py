import numpy as np
from numba import cuda
import math

# NOTES
# Block size (number of threads per block) is often crucial.
# From a software perspective, the number of threads per block
# determines how many threads are given an area of shared memory.
# From a hardware perspective, the number of threads must be large
# enough for full occupation of execution units.
# See https://docs.nvidia.com/cuda/cuda-c-programming-guide/


# serial version
def matmul_cpu(A, B, C):
    # A is n x m
    # B is m x p
    # C will be n x p
    n = len(A)
    m = len(B)
    p = len(B[0])

    for i in range(n):
        for j in range(p):
            total = 0
            for k in range(m):
                total += (A[i][k] * B[k][j])
            C[i][j] = total


# CUDA Kernel
@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    n = C.shape[0]
    m = B.shape[0]
    p = C.shape[1]
    if i < n and j < p:
        tmp = 0.0
        for k in range(m):
            tmp += (A[i, k] * B[k, j])
        C[i, j] = tmp


def manual_setup(A, B, C, nthreads=16):
    threads_per_block = (nthreads, nthreads)
    blocks_per_grid_x = math.ceil(C.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(C.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    matmul[blocks_per_grid, threads_per_block](A, B, C)


if __name__ == "__main__":
    N = 1000
    A = np.random.random((N, N))
    B = np.random.random((N, N))
    C = np.zeros_like(A)

    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array_like(A_gpu)

    manual_setup(A_gpu, B_gpu, C_gpu)
    C = C_gpu.copy_to_host()
    print(C.mean())
    print((A @ B).mean())
