import pytest

import logging

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask
import dask_cudf

import cudf
import numba.cuda as cuda
import numpy as np

from tornado import gen

import pandas.testing

from dask_cuml import knn as cumlKNN
from dask_cuml import core

def test_end_to_end():

    cluster = LocalCUDACluster(threads_per_worker=10)
    client = Client(cluster)


    def create_df(n):
        ret = cudf.DataFrame([(str(i), np.random.uniform(-1, 1, 25).astype(np.float32))
                            for i in range(5)])


        return ret

    dfs = [client.submit(create_df, n) for n in range(10)]

    print(str(dfs))

    X_df = dask_cudf.from_delayed(dfs).persist()

    print(str(client.who_has()))

    print("X: " + str(X_df.compute()))

    lr = cumlKNN.KNN()
    lr.fit(X_df)

    I, D = lr.kneighbors(X_df, 2)

    print("D: " + str(D.compute()))
    print("I: " + str(I.compute()))

    assert(0==1)

    cluster.close()