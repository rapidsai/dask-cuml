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
from .core import *

from threading import Lock, Thread

# from cuml import LinearRegression as cumlLinearRegression
from cuml import ols_spmg as cuOLS
import cuml
import logging

import itertools

import random

from toolz import first, assoc


from tornado import gen
import dask_cudf, cudf

import logging

import numba.cuda

import time

from dask.distributed import get_worker, get_client, Client

from dask import delayed
from collections import defaultdict
from dask.distributed import wait, default_client
import dask.dataframe as dd
import dask.array as da

import numpy as np

import os


def _fit(dfs, params):
    """
    This performs the actual MG fit logic.
    :param [(X.__cuda_array_interface__, y.__cuda_array_interface__)]:
        list of tuples of __cuda_array_interface__ dicts for X & y values
    :param params
        dict containing input parameters (fit_intercept,normalize,algo)
    :returns
        The resulting coef_ and intercept_ values (in that order)
    """

    print("CURRENT DEVICE: " + str(numba.cuda.get_current_device().id))

    print("DFS in FIT: " + str(dfs))

    print(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    try:
        ols = cuOLS()

        print(params)

        ols.fit(dfs, params)

        #TODO: remove return, not needed since we are passing the pointers to coeffs in params
        ret = (cudf.Series([1, 2, 3, 4]), 5)
        return ret
    except Exception as e:
        print("FAILURE in FIT: " + str(e))


def _predict(X_dfs, coeff_ptr, intercept, params):
    """
    This performs the actual MG predict logic. It should
        1. Create an empty cudf to hold the resulting predictions
        2. Make call to cython function with list of cudfs & coeff_ptrs
        3. Return cudf with resulting predictions

    The resulting predictions can be combined back into a dask-cudf. Since order
    matters here, it's important that the resulting predictions be returned
    as a list of cudfs (or a single cudf) ordered the same as the input cudfs.

    A list of pointers to the coeff data across gpus will be maintained in threads and
    closed when this computation is complete. This is only necessary until the distributed
    GEMM operation supports one process per GPU.

    :param X_df:
        cudf object to predict
    :param coeff_ptr
        a list of dicts following the __cuda_aray_interface__ format
    :return:
        cudf containing predictions
    """
    print("ALLOC_INFO: " + str(X_dfs))

    return cudf.Series([1, 2, 3, 4, 5])


def _predict_on_worker(data, params):
    coeffs, intercept, ipcs, devarrs = data


    print(ipcs)

    dev_ipcs = defaultdict(list)
    [dev_ipcs[dev].append(p) for p, dev in ipcs]

    open_ipcs = [new_ipc_thread(p, dev) for dev, p in dev_ipcs.items()]
    print(open_ipcs)

    alloc_info = list(itertools.chain([t.info() for t in open_ipcs]))
    print(alloc_info)
    alloc_info.extend([build_alloc_info(t) for t, dev in devarrs])

    try:
        # Call _predict() w/ all the cudfs on this worker and our coefficient pointers
        m = _predict(alloc_info, coeffs, intercept, params)

        [t.close() for t in open_ipcs]
        [t.join() for t in open_ipcs]

        return m

    except Exception as e:
        print("Failure: " + str(e))


def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)


def _fit_on_worker(data, params):

    ipc_dev_list, devarrs_dev_list = data

    print(":::::::::::::::::::::::fit_on_worker")

    print(ipc_dev_list)


    #TODO: One ipc thread per device instead of per x,y,coef tuple
    open_ipcs = []
    for p, dev in ipc_dev_list:
        for x, y, coef in p:
            ipct = new_ipc_thread([x, y, coef], dev)
            open_ipcs.append(ipct)

    alloc_info = list(itertools.chain([t.info() for t in open_ipcs]))
    alloc_info.extend(
        list(itertools.chain(
            [[(build_alloc_info(X)[0], build_alloc_info(y)[0], build_alloc_info(coef)[0]) for X,y,coef in p]
             for p, dev in devarrs_dev_list])))

    # Call _fit() w/ all the cudfs on this worker and our coefficient pointers
    m = _fit(alloc_info, params)

    [t.close() for t in open_ipcs]
    [t.join() for t in open_ipcs]

    return m


def build_alloc_info(p): return [p.__cuda_array_interface__]


def get_ipc_handles(arr):
    return arr[0].get_ipc_handle(), arr[1]


def get_input_ipc_handles(arr):

    arrs, dev = arr
    ret = [(X.get_ipc_handle(), y.get_ipc_handle(), coef.get_ipc_handle()) for X, y, coef in arrs]

    return ret, dev


def as_gpu_matrix(arr):
    mat = arr.as_gpu_matrix(order="F")
    dev = device_of_devicendarray(mat)

    print("DEVICE::::::::" + dev)

    # Return canonical device id as string
    return mat, dev


def to_gpu_array(arr):

    mat = arr.to_gpu_array()
    dev = device_of_devicendarray(mat)

    # Return canonical device id as string
    return mat, dev


def inputs_to_device_arrays(arr):
    """
    :param arr:
        A tuple in the form of (X, y)
    :return:
    """

    mats = [(X.as_gpu_matrix(order="F"), y.to_gpu_array(), coef.to_gpu_array()) for X, y, coef in arr]
    dev = device_of_devicendarray(mats[0][0])

    # Return canonical device id as string
    return mats, dev


def extract_part(data, part):
    return data[part]


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

    @staticmethod
    def _get_algorithm_int(algorithm):
        return {
            'svd': 0,
            'eig': 1
        }[algorithm]

    def _build_params_map(self):
        return {"fit_intercept": self.fit_intercept, "normalize":self.normalize, "algo": self.algo}

    @gen.coroutine
    def _do_fit(self, X_df, y_df):

        coefs = cudf.Series(np.zeros(X_df.shape[1]))
        print(coefs)

        # Creating the coefs as a distributed cudf
        self.coef_ = dask_cudf.from_cudf(coefs, npartitions=2).persist()
        client = default_client()

        # Break apart Dask.array/dataframe into chunks/parts
        data_parts = X_df.to_delayed()
        label_parts = y_df.to_delayed()
        coef_parts = self.coef_.to_delayed()
        if isinstance(data_parts, np.ndarray):
            assert data_parts.shape[1] == 1
            data_parts = data_parts.flatten().tolist()
        if isinstance(label_parts, np.ndarray):
            assert label_parts.ndim == 1 or label_parts.shape[1] == 1
            label_parts = label_parts.flatten().tolist()

        # Arrange parts into pairs.  This enforces co-locality
        parts = list(map(delayed, zip(data_parts, label_parts, coef_parts)))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)

        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        who_has = yield client.who_has(parts)

        worker_parts = {}
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])

        """
        Create IP Handles on each worker hosting input data
        """

        # Format of input_devarrays = ([(X, y)..], dev)
        input_devarrays = [(worker, client.submit(inputs_to_device_arrays, part, workers=[worker]))
                    for worker, part in worker_parts.items()]

        yield wait(input_devarrays)

        """
        Gather IPC handles for each worker and call _fit() on each worker containing data.
        """
        res = []

        exec_node = input_devarrays[0][0]

        print("HERE:::::::::::::::::::::::::::::::::::::::")
        print(input_devarrays)

        # Need to fetch coefficient parts on worker
        on_worker = list(filter(lambda x: x[0] == exec_node, input_devarrays))
        print(on_worker)
        not_on_worker = list(filter(lambda x: x[0] != exec_node, input_devarrays))
        print(not_on_worker)

        ipc_handles = [client.submit(get_input_ipc_handles, future, workers=[a_worker])
                       for a_worker, future in not_on_worker]

        raw_arrays = [future for a_worker, future in on_worker]


        print("HERE:::::::::::::::::::::::::::::::::::::::")

        # IPC Handles are loaded in separate threads on worker so they can be
        # used to make calls through cython

        ret = client.submit(_fit_on_worker, (ipc_handles, raw_arrays),
                                                  self._build_params_map(), workers=[exec_node])

        yield wait(ret)

        # We can assume a single coeff array and intercept for now.
        # coeffs = (worker, client.submit(extract_part, ret, 0, workers = [worker]))
        intercept = client.submit(extract_part, ret, 1, workers = [worker])

        raise gen.Return((coeffs, intercept))

    def fit(self, X_df, y_df):
        """
        Fits a multi-gpu linear regression model such that each the resulting coefficients are
        also distributed across the GPUs.
        :param futures:
        :return:
        """

        client = default_client()

        # Coeffs should be a future with a handle on a Dataframe on a single worker.
        # Intercept should be a future with a handle on a float on a single worker.
        coeffs, intercepts = client.sync(self._do_fit, X_df, y_df)

        self.coef_ = coeffs
        self.intercept_ = intercepts

    @gen.coroutine
    def _do_predict(self, dfs):

        client = default_client()

        # Break apart Dask.array/dataframe into chunks/parts
        data_parts = dfs.to_delayed()
        if isinstance(data_parts, np.ndarray):
            assert data_parts.shape[1] == 1
            data_parts = data_parts.flatten().tolist()

        # Arrange parts into pairs.  This enforces co-locality
        parts = list(map(delayed, data_parts))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)

        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        who_has = yield client.who_has(parts)

        worker_parts = []
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            worker_parts.append((worker, key_to_part_dict[key]))

        """
        Build Numba devicearrays for all coefficient chunks
        """

        # Have worker running the coeffs execute the predict logic
        exec_node, coeff_future = self.coef_

        gpu_data = [(worker, client.submit(as_gpu_matrix, part, workers = [worker]))
                    for worker, part in worker_parts]

        # build ipc handles
        gpu_data_excl_worker = list(filter(lambda d: d[0] != exec_node, gpu_data))
        gpu_data_incl_worker = list(filter(lambda d: d[0] == exec_node, gpu_data))

        ipc_handles = [client.submit(get_ipc_handles, future, workers=[worker])
                       for worker, future in gpu_data_excl_worker]

        raw_arrays = [future for worker, future in gpu_data_incl_worker]

        f = client.submit(_predict_on_worker,
                          (coeff_future, self.intercept_, ipc_handles, raw_arrays),
                          self._build_params_map(),
                          workers=[exec_node])

        yield wait(f)


        # Turn resulting cudf future into a dask-cudf and return it. For now, returning the futures
        # pointing to the data.

        yield wait(f)
        return gen.Return(f)

    def predict(self, X):
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

        client = default_client()
        return client.sync(self._do_predict, X).value

    def _build_host_dict(self, gpu_futures, client):

        who_has = client.who_has(gpu_futures)

        key_to_host_dict = {}
        for key in who_has:
            key_to_host_dict[key] = parse_host_port(who_has[key][0])

        hosts_to_key_dict = {}
        for key, host in key_to_host_dict.items():
            if host not in hosts_to_key_dict:
                hosts_to_key_dict[host] = set([key])
            else:
                hosts_to_key_dict[host].add(key)

        workers = [key[0] for key in list(who_has.values())]
        return build_host_dict(workers)
