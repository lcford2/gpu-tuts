import numpy as np
from numba import cuda

# NOTES
# Block size (number of threads per block) is often crucial.
# From a software perspective, the number of threads per block
# determines how many threads are given an area of shared memory.
# From a hardware perspective, the number of threads must be large
# enough for full occupation of execution units.
# See https://docs.nvidia.com/cuda/cuda-c-programming-guide/


# serial version
def vec_add_cpu(a, b, c):
    size = len(c)
    for i in range(size):
        c[i] = a[i] + b[i]


# CUDA Kernel
@cuda.jit
def vec_add(a, b, c):
    thread_id = cuda.grid(1)
    size = len(c)

    if thread_id < size:
        c[thread_id] = a[thread_id] + b[thread_id]


def auto_setup(a, b, c):
    # create appropriate launch conditions for 1d kernel
    vec_add.forall(len(a))(a, b, c)
    print(c.copy_to_host().mean())


def manual_setup(a, b, c, nthreads=128):
    # you can also configure the grid manually
    # nthreads is the number of threads per block
    # a warp is 32 threads and we want
    # several warps per block

    # enough blocks to cover the array
    nblocks = (len(a) // nthreads) + 1
    vec_add[nblocks, nthreads](a, b, c)
    print(c.copy_to_host().mean())


if __name__ == "__main__":
    N = 100000
    A = np.random.random(N)
    B = np.random.random(N)
    C = np.zeros_like(A)

    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array_like(A_gpu)

    auto_setup(A_gpu, B_gpu, C_gpu)
    manual_setup(A_gpu, B_gpu, C_gpu)
