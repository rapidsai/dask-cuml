# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from .core import parse_host_port, IPCThread, build_host_dict, select_device

from cuml import LinearRegression as cumlLinearRegression
import logging

import random

from tornado import gen
import dask_cudf, cudf

import logging

import time

from dask.distributed import get_worker, get_client, Client

from dask import delayed
from collections import defaultdict
from dask.distributed import wait, default_client
import dask.dataframe as dd
import dask.array as da

import numpy as np

def _fit(dfs):
    """
    This performs the actual MG fit logic. It should
        1. Create an empty numba array to hold the resulting coefficients
        2. Make call to cython function with list of cudfs
        3. Return resulting coefficients.
    :param (X_cudf, y_cudf):
        list of Numba device ndarray objects
    """

    pass

def _predict(X_df):
    """
    This performs the actual MG predict logic. It should
        1. Create an empty cudf to hold the resulting coefficients
        2. Make call to cython function with list of cudfs
        3. Return cudf with resulting predictions

    The resulting predictions can be combined back into a dask-cudf. Since order
    matters here, we let dask manage the resulting futures and construct the dask-cudf.
    :param X_df:
        cudf object to predict
    :return:
        cudf containing predictions
    """
    pass

class LinearRegression(object):
    """
    Model-Parallel Multi-GPU Linear Regression Model.

    Data is spread across Dask workers in a one worker per GPU fashion. MPI is used in the C++
    algorithms layer in order to share data & state when necessary.
    """
    def __init__(self, algorithm='eig', fit_intercept=True, normalize=False):

        """
        Initializes the liner regression class.

        Parameters
        ----------
        algorithm : Type: string. 'eig' (default) and 'svd' are supported algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        if algorithm in ['svd', 'eig']:
            self.algo = self._get_algorithm_int(algorithm)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        self.intercept_value = 0.0



    def fit(self, X_df, y_df):
        """
        Fits a multi-gpu linear regression model such that each the resulting coefficients are
        also distributed across the GPUs.
        :param futures:
        :return:
        """
        client = default_client()

        # Break apart Dask.array/dataframe into chunks/parts
        data_parts = X_df.to_delayed()
        label_parts = y_df.to_delayed()
        if isinstance(data_parts, np.ndarray):
            assert data_parts.shape[1] == 1
            data_parts = data_parts.flatten().tolist()
        if isinstance(label_parts, np.ndarray):
            assert label_parts.ndim == 1 or label_parts.shape[1] == 1
            label_parts = label_parts.flatten().tolist()

        # Arrange parts into pairs.  This enforces co-locality
        parts = list(map(delayed, zip(data_parts, label_parts)))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)

        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally        gpu_futures = client.sync(self._get_mg_info,input_data)

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        # Dask-cudf should be one process per GPU. We assume single-node Dask cluster for first iteration.
        worker_part_map = client.has_what()

        self.coeffs = [client.submit(_fit, [p for p in key_to_part_dict[keys]], workers = [worker])
                       for worker, keys in worker_part_map.items()]

        wait(self.coeffs)

    def predict(self, df):
        """
        Predict values for the multi-gpu linear regression model by making calls to the predict function
        with dask-cudf objects. This gets a little more complicated since we will need to collect the
        coeff pointers onto each worker to pass into the c++ layer. This has the following steps:

            1. Collect coefficient pointers onto each worker that has data in the input dask-cudf
            2. Have each worker call down to the c++ predict function with their coefficient pointers
               and X values to predict.
        :param df:
            a dask-cudf with data distributed one worker per GPU
        :return:
            a dask-cudf containing outputs of the linear regression
        """



        pass
