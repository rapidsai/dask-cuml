import pytest

import logging

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

import dask_cudf

import cudf
import numba.cuda as cuda
import numpy as np

import pandas.testing

from dask_cuml import linear_regression as cumlLR
from dask_cuml import core

def test_end_to_end():

    cluster = LocalCUDACluster(threads_per_worker=10)
    client = Client(cluster)

    X = cudf.DataFrame([('a', [0, 1, 2, 3, 4])])
    y = cudf.Series([0, 1, 2, 3, 4])

    X_df = dask_cudf.from_cudf(X, chunksize=1).persist()
    y_df = dask_cudf.from_cudf(y, chunksize=1).persist()

    lr = cumlLR.LinearRegression()
    lr.fit(X_df, y_df)

    assert(lr.intercept_.result() == 5)

    actual_coeffs = lr.coef_[1].result().to_pandas()
    expected_coeffs = cudf.Series([1, 2, 3, 4]).to_pandas()

    pandas.testing.assert_series_equal(actual_coeffs, expected_coeffs)

    g = lr.predict(X_df)

    actual_result = g.result().to_pandas()
    expected_result = cudf.Series([1, 2, 3, 4, 5]).to_pandas()

    pandas.testing.assert_series_equal(actual_result, expected_result)

    cluster.close()