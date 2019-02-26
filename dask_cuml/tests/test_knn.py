import pytest

import logging

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

import dask_cudf

import cudf
import numba.cuda as cuda
import numpy as np

import pandas.testing

from dask_cuml import knn as cumlKNN
from dask_cuml import core


def test_end_to_end():

    cluster = LocalCUDACluster(threads_per_worker=10)
    client = Client(cluster)

    X = cudf.DataFrame([('a', np.array([0, 1, 2, 3, 4], dtype=np.float32)),
                        ('b', np.array([5, 6, 7, 7, 8], dtype=np.float32))])

    X_df = dask_cudf.from_cudf(X, npartitions=2).persist()

    print("X: " + str(X_df.compute()))

    lr = cumlKNN.KNN()
    lr.fit(X_df)

    I, D = lr.kneighbors(X_df, 2)

    parts = I.to_delayed()
    who_has = client.who_has(parts)

    print("D: " + str(D.compute()))
    print("I: " + str(I.compute()))

    assert(0==1)

    cluster.close()