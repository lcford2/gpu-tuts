from time import perf_counter as timer

import numpy as np
import pycuda.autoinit
import pycuda.driver
import pycuda.gpuarray


def gpuarray_simple(size):
    data_size = 1 << size  # same as 2 ** 25
    factor = 10 * np.random.random()

    host_array = np.random.uniform(0, 50, size=data_size)

    # GPUs use 32 bit FP numbers
    host_array = host_array.astype(np.float32)

    # create device array from host_array
    data_move_time1 = timer()
    device_array = pycuda.gpuarray.to_gpu(host_array)
    data_move_time2 = timer()
    dtime1 = timer()
    device_results = factor * device_array
    dtime2 = timer()

    # get results
    data_move_time3 = timer()
    host_results = device_results.get()
    data_move_time4 = timer()

    device_calc_time = dtime2 - dtime1
    device_data_time = (data_move_time4 - data_move_time3) + (
        data_move_time2 - data_move_time1
    )
    device_total_time = device_calc_time + device_data_time

    htime1 = timer()
    host_only_results = factor * host_array
    htime2 = timer()
    host_time = htime2 - htime1

    assert np.allclose(host_results, host_only_results), "gpuarray_simple: Failed"

    # print(f"GPU Time ({size}): {device_time:.4f}; CPU Time ({size}): {host_time:.4f}")
    print(
        f"{data_size:15} | {device_data_time:.4f} | {device_calc_time:.4f} |"
        f" {device_total_time:.4f} | {host_time:.4f}"
    )


if __name__ == "__main__":
    print(
        "{:^15s} | {:6s} | {:6s} | {:6s} | {:6s}".format(
            "Data Size", "DTtime", "CLtime", "TLtime", "HTtime"
        )
    )
    for i in range(20, 25):
        gpuarray_simple(i)
