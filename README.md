# gpu-tuts

## General Purpose GPU Programming Notes

To view a collection of high level notes on General Purpose GPU programming please see the `gpu-notes.ipynb` notebook in the root of this repository.

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
  - CuPy is an array library that aims to mimic NumPy's and SciPy's API for GPU accelerated computing with Python
  - It leverages CUDA and CUDA accelerated libraries like cuBLAS, cuRAND, cuSOLVER, and others. 
  - In most cases it can be a drop in replacement for numpy.
    - Instead of `np.array` use `cp.array`
    - Instead of `from scipy.fft import fft` use `from cupyx.scipy.fft import fft` 
  - This is the library that many of the [Rapids AI](https://rapids.ai/) libraries are built on. 
  - It is probably the easiest method to begin writing GPU accelerated code in python
  - You can find a cupy example in the `examples/python/cupy' directory

- [Rapids AI](https://rapids.ai/)
  - A suite of open source libraries and APIs designed to facilitate end-to-end data science pipelines entirely on GPUs
  - Rapids provides:
    - [cuDF](https://docs.rapids.ai/api/cudf/stable/) - A near drop-in replacement for pandas that allows dataframe manipulation and storage directly on GPUs. You can find an example using cuDF at `examples/python/rapids/cudf`
    - [cuML](https://docs.rapids.ai/api/cuml/stable/) - A GPU accelerated machine learning library that attempts to mimic the API of scikit-learn. You can find an example using cuML at `examples/python/rapids/cuml`.
    - [cuGraph](https://docs.rapids.ai/api/cugraph/stable/) - GPU accelerated graph algorithms that operates on cuDF dataframes. Attempts to mimic the API of NetworkX (the standard graph library in python). An example using cuGraph can be found at `examples/python/rapids/cugraph`
    - [cuSpatial](https://docs.rapids.ai/api/cuspatial/stable/) - GPU accelerated library for spatial index and join functions. Integrates well with GeoPandas (a common geodata library in python that mimics pandas API). GPUs are well suited to spatial data analytics as you are often applying the same calculations over and over again on gridded data.
    - [cuSignal](https://docs.rapids.ai/api/cusignal/stable/) - GPU accelerated signal processing that attempts to be a direct port of Scipy Signal
    - [cuCIM](https://docs.rapids.ai/api/cucim/stable/) - Computer vision and image processing software library for multidimensional images (e.g. biomedical, geospatial, remote sensing, etc...). Attempts to mirror the scikit-image API. 
