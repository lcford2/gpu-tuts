from time import perf_counter as timer

import cudf as cd
import numpy as np
import pandas as pd
from colorama import Fore, Style
from colorama import init as colorama_init
from numba import cuda

colorama_init()

CPU_STRING = Fore.BLUE + "CPU" + Style.RESET_ALL
GPU_STRING = Fore.GREEN + "GPU" + Style.RESET_ALL


def my_square(x):
    return x * x


def cpu_square(df):
    s = timer()
    rel_squared = df["release_acre-feet-per-day"].apply(my_square)
    e = timer()
    print(rel_squared.mean())
    print(f"{CPU_STRING} - Describe Time: {e - s:.4f} seconds")


def gpu_square(df):
    # call it once to compile the function
    rel_squared = df["release_acre-feet-per-day"].apply(my_square)
    # call it again with the compiled function
    # future calls will use the cached compiled version
    s = timer()
    rel_squared = df["release_acre-feet-per-day"].apply(my_square)
    e = timer()
    print(rel_squared.mean())
    print(f"{GPU_STRING} - Describe Time: {e - s:.4f} seconds")


def calc_mass_balance(storage, storage_pre, release, inflow, mb, thread):
    iterator = enumerate(zip(storage, storage_pre, release, inflow))
    for i, (st, stp, rel, inf) in iterator:
        mb[i] = -st + stp - rel + inf
        thread[i] = cuda.threadIdx.x


@cuda.jit
def calc_mass_balance_kernel(storage, storage_pre, release, inflow, mb, thread):
    i = cuda.grid(1)
    if i < storage.size:
        mb[i] = -storage[i] + storage_pre[i] - release[i] + inflow[i]
        thread[i] = cuda.threadIdx.x


if __name__ == "__main__":
    # traditional way to load data with pandas
    hdf = pd.read_csv("../../../data/csv/reservoir_data.csv")
    hdf["storage-pre_acre-feet"] = hdf.groupby("site-name")["storage_acre-feet"].shift(1)
    hdf = hdf.dropna()

    # load data with cuDF
    ddf = cd.read_csv("../../../data/csv/reservoir_data.csv")
    ddf["storage-pre_acre-feet"] = ddf.groupby("site-name")["storage_acre-feet"].shift(1)
    ddf = ddf.dropna()

    cpu_square(hdf)
    print()
    gpu_square(ddf)
    print()

    s = timer()
    d_out = ddf.apply_rows(
        calc_mass_balance,
        incols={
            "storage_acre-feet": "storage",
            "storage-pre_acre-feet": "storage_pre",
            "release_acre-feet-per-day": "release",
            "net-inflow_acre-feet-per-day": "inflow",
        },
        outcols={"mb": np.float32, "thread": np.int32},
        kwargs={},
    )
    e = timer()
    print(f"Apply rows time: {e - s:.4f}")

    s = timer()
    d_out = ddf.apply_rows(
        calc_mass_balance,
        incols={
            "storage_acre-feet": "storage",
            "storage-pre_acre-feet": "storage_pre",
            "release_acre-feet-per-day": "release",
            "net-inflow_acre-feet-per-day": "inflow",
        },
        outcols={"mb": np.float32, "thread": np.int32},
        kwargs={},
    )
    e = timer()
    print(f"Apply rows time (2): {e - s:.4f}")

    ddf["mb"] = np.zeros_like(ddf["storage_acre-feet"], dtype=np.float64)
    ddf["thread"] = np.zeros_like(ddf["storage_acre-feet"], dtype=np.int32)

    s = timer()
    calc_mass_balance_kernel.forall(ddf["storage_acre-feet"].size)(
        ddf["storage_acre-feet"].values,
        ddf["storage-pre_acre-feet"].values,
        ddf["release_acre-feet-per-day"].values,
        ddf["net-inflow_acre-feet-per-day"].values,
        ddf["mb"].values,
        ddf["thread"].values,
    )
    e = timer()
    print(f"For all time: {e - s:.4f}")

    s = timer()
    calc_mass_balance_kernel.forall(ddf["storage_acre-feet"].size)(
        ddf["storage_acre-feet"].values,
        ddf["storage-pre_acre-feet"].values,
        ddf["release_acre-feet-per-day"].values,
        ddf["net-inflow_acre-feet-per-day"].values,
        ddf["mb"].values,
        ddf["thread"].values,
    )
    e = timer()
    print(f"For all time (2): {e - s:.4f}")
