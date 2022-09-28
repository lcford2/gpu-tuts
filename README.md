# gpu-tuts

## Environment setup

Most of the examples in this repository are designed to be run on Henry2. With that in mind, a `conda` environment was created that can be used when launching jobs with `bsub`. 
This environment can be activated by running `conda activate /usr/local/usrapps/ce791/conda_envs/gpu-env` after [loading the `conda` module on Henry2](https://hpc.ncsu.edu/Software/Apps.php?app=Conda). 
This environment corresponds to the `gpu-env.yml` in the root folder of this repository. 
The `gpu-dev.yml` file has all the packages of `gpu-env.yml` but with some added tools for developing code within this repository.

## Python Examples

Over the past decade there has been a large push to make it easier for users to leverage the computational resources provided by GPUs. 
A large part of this effort has resulted in a handful of powerful python libraries that lower the barrier to entry for general purpose GPU programming. 
The libraries cover a wide spectrum of GPU programming applications without requiring the users to know the intimate details of GPU programming. 

In this repository I have highlighted 6 packages that provide varying levels of abstraction. 
All of these packages are built on the [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) programming model. 
Though it is not necessary to be familiar with CUDA concepts to use these packages, the better understanding you have of CUDA the more you can get out of these packages.

### GPU Python Packages
- [pyCUDA](https://documen.tician.de/pycuda/) 
  - Provides a nice pythonic interface to CUDA
  - Object cleanup that is tied to the lifetime of objects and its dependencies
  - Nice abstractions for convenience like `SourceModule` for kernels or `GPUArray` for arrays stored on the device
  - Very fast as the core functionalities are written in C++
  With pyCUDA, you must take care of many of the same tasks to perform GPU programming in CUDA.
  These tasks include moving data to and from the device (GPU) and creating functions that will run on the device.
  You can find a short example of how to use pyCUDA in the `examples/python/pycuda` directory with a submission file for Henry2.
  
- [Numba](https://numba.readthedocs.io/en/stable/cuda/index.html)
  - Numba is a just-in-time (JIT) compiler for python.
    - This means that, when told to, Numba will compile python code to machine code and then everytime that code is called again it will be **significantly** faster.
    - You can use Numba for CPU only code as well to speed up your python code. To get the best performance, you should write functions that contain your computation that is called often and then pass only nearly primitive objects to that function (integers, floats, lists, np.arrays, etc.)
  - To use Numba on GPUs: 
    - Write a python function that performs calculation on an array or matrix of data
    - Replace the iterator in that function with a CUDA thread ID (e.g., `thread_x = cuda.threadIdx.x`)
    - Add the `@cuda.jit` decorator to the function
    - Specify a computational grid when calling the function following the examples [here](https://numba.readthedocs.io/en/stable/cuda/kernels.html)
  - Numba allows you to write purely python functions that will run on GPUs, thus further lowering the barrier to entry for GPU programming.
  - You can find a Numba example in the `examples/python/numba` directory. (This example cannot run on Henry2, but you could run on your local machine if you have an NVIDIA GPU)
  

- [CuPy](https://cupy.dev/)
 
