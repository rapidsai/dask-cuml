import pytest

import logging

from dask.distributed import Client, LocalCluster, wait

import cudf
import numba.cuda as cuda
import numpy as np

from dask_cuml import mean
from dask_cuml import core

def func(x):
    return x + 1


def create_cudf(dev):
    cuda.select_device(dev)
    logging.debug("Creating dataframe on device " + str(dev))
    return (dev, cudf.DataFrame(
        [('a', np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)),
         ('b', np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32))]
    ))


def test_answer():
    cluster = LocalCluster(n_workers = 2, threads_per_worker = 5, processes = True)
    client = Client(cluster)

    assignments, workers = core.assign_gpus()

    res = [client.submit(create_cudf, future, workers = [worker]) for future, worker in zip(assignments, workers)]
    wait(res)

    m = mean.MGMean()
    ret = m.calculate(res)

    print(str(ret))

    client.close()

    assert (1 == 2)
