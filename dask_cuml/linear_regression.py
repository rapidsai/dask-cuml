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

def _fit(dfs):
    """
    This performs the actual MG fit logic. It should
        1. Create an empty numba array to hold the resulting coefficients
        2. Make call to cython function with list of cudfs
        3. Return resulting coefficients.
    :param [(X_cudf, y_cudf)]:
        list of tuples of __cuda_device_array__ dicts for X & y values
    """

    print("CURRENT DEVICE: " + str(numba.cuda.get_current_device().id))

    try:
        # TODO: Using Series for coeffs throws an error after the 2nd or third training of a model
        # The error is related to the bitmask or pickle serialization. Very strange,
        # considering there shouldn't be any pickling in this line.
        ret = cudf.Series([1, 2, 3, 4])
        return ret
    except Exception as e:
        print("FAILURE in FIT: " + str(e))


def _predict(X_dfs, coeff_ptrs):
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
    :param coeff_ptrs
        a list of dicts following the __cuda_aray_interface__ format
    :return:
        cudf containing predictions
    """

    print("Called _predict()")

    return cudf.Series([1, 2, 3, 4, 5])

def _predict_on_worker(data):
    coeffs, ipcs, devarrs = data

    print("PREDICT IPCS: "+ str(ipcs))
    print("PREDICT DEVARRS: " + str(devarrs))

    dev_ipcs = {}
    for p, dev in ipcs:
        if dev not in dev_ipcs:
            dev_ipcs[dev] = []
        dev_ipcs[dev].append(p)

    open_ipcs = [new_ipc_thread(p, dev) for dev, p in dev_ipcs.items()]

    alloc_info = [t.info() for t in open_ipcs]
    alloc_info.extend([build_alloc_info(t) for t, dev in devarrs])

    try:
        # Call _predict() w/ all the cudfs on this worker and our coefficient pointers
        m = _predict(coeffs, alloc_info)

        print("Returned from PREDICT")

        return open_ipcs, devarrs, m

    except Exception as e:
        print("Failure: " + str(e))

def close_threads(d):
    ipc_threads, devarrs, result = d
    [t.close() for t in ipc_threads]

def join_threads(d):
    ipc_threads, devarrs, result = d
    [t.join() for t in ipc_threads]


def get_result(d):

    print("get_result:  " + str(d))

    ipc_threads, devarrs, result = d
    return result

def group(lst, n):
  for i in range(0, len(lst), n):
    val = lst[i:i+n]
    if len(val) == n:
      yield tuple(val)



def _fit_on_worker(data):

    ipc_dev_list, devarrs_dev_list = data

    print("DEV_ARRS_LIST: " + str(devarrs_dev_list))

    open_ipcs = [new_ipc_thread(itertools.chain([[X,y] for X,y in p]), dev) for p, dev in ipc_dev_list]

    alloc_info = [group(t.info(), 2) for t in open_ipcs]

    print(str("ALLOC INFO: " + str(alloc_info)))

    alloc_info.extend(
        list(itertools.chain(
            [[(build_alloc_info(X)[0], build_alloc_info(y)[0]) for X,y in p]
             for p, dev in devarrs_dev_list])))

    # Call _predict() w/ all the cudfs on this worker and our coefficient pointers
    m = _fit(alloc_info)

    return open_ipcs, devarrs_dev_list, m


def build_alloc_info(p): return [p.__cuda_array_interface__]


def get_ipc_handles(arr):


    return arr[0].get_ipc_handle(), arr[1]


def get_input_ipc_handles(arr):

    print("input_ipc_handles: " + str(arr))

    arrs, dev = arr

    ret = [(X.get_ipc_handle(), y.get_ipc_handle()) for X, y in arrs]

    return ret, dev


def as_gpu_matrix(arr):
    mat = arr.as_gpu_matrix(order="F")

    import os
    dev = cuml.device_of_ptr(mat.device_ctypes_pointer.value)
    print("dev_of_ptr: " + str(dev))
    print("DEVICE: " + str(numba.cuda.get_current_device()))
    return mat, dev


def to_gpu_array(arr):

    print("ARR: "+ str(arr))

    mat = arr.to_gpu_array()

    import os
    dev = cuml.device_of_ptr(mat.device_ctypes_pointer.value)
    print("dev_of_ptr: " + str(dev))
    print("DEVICE: " + str(numba.cuda.get_current_device()))
    return mat, dev


def inputs_to_device_arrays(arr):
    """
    :param arr:
        A tuple in the form of (X, y)
    :return:
    """

    print("HANDLES: "+ str(arr))

    mats = [(X.as_gpu_matrix(order="F"), y.to_gpu_array()) for X, y in arr]

    # Both X & y should be on the same device at this point (by being on the same worker)
    dev = cuml.device_of_ptr(mats[0][0].device_ctypes_pointer.value)
    print("dev_of_ptr: " + str(dev))
    print("DEVICE: " + str(numba.cuda.get_current_device()))

    print("MATS: " + str(mats))
    print("dev=" + str(dev))

    return mats, dev


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
        self.coeffs = None;


    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1
        }[algorithm]


    @gen.coroutine
    def _do_fit(self, X_df, y_df):


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

        print("input_devarrays: " + str(input_devarrays))

        """
        Gather IPC handles for each worker and call _fit() on each worker containing data.
        """
        worker_results = {}
        res = []

        exec_node = input_devarrays[0][0]

        print("exec_node: "+ str(exec_node))

        # Need to fetch coefficient parts on worker
        on_worker = list(filter(lambda x: x[0] == exec_node, input_devarrays))
        not_on_worker = list(filter(lambda x: x[0] != exec_node, input_devarrays))

        ipc_handles = [client.submit(get_input_ipc_handles, future, workers=[a_worker])
                       for a_worker, future in not_on_worker]

        raw_arrays = [future for a_worker, future in on_worker]

        print("ipc_handles: "+ str(ipc_handles))
        print("raw_arrays: " + str(raw_arrays))

        # IPC Handles are loaded in separate threads on worker so they can be
        # used to make calls through cython
        worker_results[exec_node] = client.submit(_fit_on_worker,
                                                (ipc_handles, raw_arrays), workers=[exec_node])

        res.append(worker_results[exec_node])

        yield wait(res)

        d = [client.submit(close_threads, future) for future in res]
        yield wait(d)

        d = [client.submit(join_threads, future) for future in res]
        yield wait(d)

        ret = [(worker, client.submit(get_result, futures, workers= [worker]))
               for worker, futures in worker_results.items()]

        yield wait(ret)
        raise gen.Return(ret)

    def fit(self, X_df, y_df):
        """
        Fits a multi-gpu linear regression model such that each the resulting coefficients are
        also distributed across the GPUs.
        :param futures:
        :return:
        """

        client = default_client()

        # Coeffs should be a future with a handle on a Dataframe on a single worker.
        self.coeffs = client.sync(self._do_fit, X_df, y_df)[0]

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


        print("WORKER PARTS: " + str(worker_parts))

        """
        Build Numba devicearrays for all coefficient chunks        
        """

        # Have worker running the coeffs execute the predict logic
        exec_node, coeff_future = self.coeffs

        gpu_data = [(worker, client.submit(as_gpu_matrix, part, workers = [worker])) for worker, part in worker_parts]

        # build ipc handles
        gpu_data_excl_worker = list(filter(lambda d: d[0] != exec_node, gpu_data))
        gpu_data_incl_worker = list(filter(lambda d: d[0] == exec_node, gpu_data))

        print("ON WORKER: " + str(len(list(gpu_data_incl_worker))))
        print("NOT ON WORKER: " + str(len(list(gpu_data_excl_worker))))

        ipc_handles = [client.submit(get_ipc_handles, future, workers=[worker])
                       for worker, future in gpu_data_excl_worker]

        raw_arrays = [future for worker, future in gpu_data_incl_worker]

        print("IPCHANDLES = " + str(ipc_handles))
        print("RAW_ARRAYS=" + str(raw_arrays))

        f = client.submit(_predict_on_worker, (coeff_future, ipc_handles, raw_arrays), workers=[exec_node])

        yield wait(f)

        print("f=" + str(f))

        # We can safely close our local threads now that the c++ API is holding onto
        # its own resources.

        d = client.submit(close_threads, f)
        yield wait(d)

        d = client.submit(join_threads, f)
        yield wait(d)

        # Turn resulting cudf future into a dask-cudf and return it. For now, returning the futures
        # pointing to the data.
        ret = client.submit(get_result, f)

        yield wait(ret)
        return gen.Return(ret)

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
        return client.sync(self._do_predict, X)

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
