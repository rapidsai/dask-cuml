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

    X = cudf.DataFrame([('a', [0, 1, 2, 3, 4]),
                        ('b', [5, 6, 7, 7, 8])])

    X_df = dask_cudf.from_cudf(X, chunksize=1).persist()

    lr = cumlKNN.KNN()
    lr.fit(X_df)

    print(str(lr.host_masters))

    g = lr.kneighbors(X, 2)



    print("OUTPUT: " + str(g[1].result()))

    assert(1==0)

    cluster.close()