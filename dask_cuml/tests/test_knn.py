import pytest

import logging

import time

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

def test_end_to_end():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    import dask
    import dask_cudf

    import pandas as pd

    import cudf
    import numpy as np

    import pandas.testing

    from dask_cuml import knn as cumlKNN

    def create_df(n):
        X = np.random.rand(620000, 1000)
        ret = cudf.DataFrame([(i,X[:,i].astype(np.float32)) for i in range(X.shape[1])])

        return ret

    workers = client.has_what().keys()

    start = time.time()

    dfs = [client.submit(create_df, n, workers = [worker])
           for worker, n in zip(workers, range(len(cluster.workers)))]
    wait(dfs)

    end = time.time() - start

    print("Creating data took " + str(end))

    start = time.time()

    X_df = dask_cudf.from_delayed(dfs)

    end = time.time() - start

    print("Creating dataframe took " + str(end))

    lr = cumlKNN.KNN()

    start = time.time()
    lr.fit(X_df)
    end = time.time() - start

    print("Fitting data took: "+ str(end))

    start = time.time()
    I, D = lr.kneighbors(X_df[0:50], 15)
    end = time.time() - start
    print("Searching data took " + str(end))


    assert(0==1)

    cluster.close()