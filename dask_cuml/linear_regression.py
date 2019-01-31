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

from threading import Lock, Thread

from cuml import LinearRegression as cumlLinearRegression
import cuml
import logging

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
        list of tuples of Numba device ndarray objects for X & y values
    """

    print("FIT CALLED: " + str(dfs))
    return cudf.Series(np.zeros(10, dtype=np.float32))

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
    print("PREDICT CALLED: " + str(X_dfs) + ", " + str(coeff_ptrs))
    return cudf.Series(np.zeros(4, dtype=np.float32))

def _predict_on_worker(data):
    parts, ipcs, devarrs = data

    def new_ipc_thread(ipc):
        t = LRIPCThread(ipc)
        t.start()
        return t

    open_ipcs = [new_ipc_thread(p) for p in ipcs]

    coeff_alloc_info = [t.info() for t in open_ipcs]
    coeff_alloc_info.extend([build_alloc_info(t[0]) for t in devarrs])

    # Call _predict() w/ all the cudfs on this worker and our coefficient pointers
    m = _predict(parts, coeff_alloc_info)

    return open_ipcs, devarrs, m


class LRIPCThread(Thread):
    """
    This mechanism gets around Numba's restriction of CUDA contexts being thread-local
    by creating a thread that can select its own device. This allows the user of IPC
    handles to open them up directly on the same device as the owner (bypassing the
    need for peer access.)
    """
    def __init__(self, ipc):

        Thread.__init__(self)

        self.lock = Lock()
        self.ipcs = [ipc[0]] # TODO: Should group by device
        self.device = ipc[1]
        self.running = False

    def run(self):

        select_device(self.device)

        print("Opening: " + str(self.device) + " " + str(numba.cuda.get_current_device()))

        self.lock.acquire()

        try:
            self.arrs = [ipc.open() for ipc in self.ipcs]

            print("IPCS: "+ str(self.ipcs))
            print("ARRS: "+ str(self.arrs))
            self.ptr_info = [x.__cuda_array_interface__ for x in self.arrs]

            self.running = True
        except Exception as e:
            logging.error("Error opening ipc_handle on device " + str(self.device) + ": " + str(e))

        self.lock.release()

        while (self.running):
            time.sleep(0.0001)

        try:
            logging.warn("Closing: " + str(self.device) + str(numba.cuda.get_current_device()))
            self.lock.acquire()
            [ipc.close() for ipc in self.ipcs]
            self.lock.release()
        except Exception as e:
            logging.error("Error closing ipc_handle on device " + str(self.device) + ": " + str(e))

    def close(self):

        """
        This should be called before calling join(). Otherwise, IPC handles may not be
        properly cleaned up.
        """
        self.lock.acquire()
        self.running = False
        self.lock.release()

    def info(self):
        """
        Warning: this method is invoked from the calling thread. Make
        sure the context in the thread reading the memory is tied to
        self.device, otherwise an expensive peer access might take
        place underneath.
        """
        while (not self.running):
            time.sleep(0.0001)

        return self.ptr_info


def build_alloc_info(p): return [p.__cuda_array_interface__]


def get_ipc_handles(arr): return arr[0].get_ipc_handle(), arr[1]


def as_gpu_matrix(arr):
    mat = arr.as_gpu_matrix(order="F")
    return mat, cuml.device_of_ptr(mat.device_ctypes_pointer.value)

def to_gpu_array(arr):
    mat = arr.to_gpu_array()
    return mat, cuml.device_of_ptr(mat.device_ctypes_pointer.value)


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
        self.coeffs = []

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1
        }[algorithm]



    @gen.coroutine
    def _do_fit(self, X_df, y_df):
        client = default_client()

        print("Co-locating data!")

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

        print(str(who_has))

        worker_parts = {}
        for key, workers in who_has.items():

            print(str(workers))
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])

        print(str(worker_parts))

        coeffs = [(worker, client.submit(_fit, keys, workers = [worker]))
                  for worker, keys in worker_parts.items()]

        yield wait(coeffs)

        raise gen.Return(coeffs)

    def fit(self, X_df, y_df):
        """
        Fits a multi-gpu linear regression model such that each the resulting coefficients are
        also distributed across the GPUs.
        :param futures:
        :return:
        """

        client = default_client()
        self.coeffs = client.sync(self._do_fit, X_df, y_df)

    @gen.coroutine
    def _do_predict(self, df):

        client = default_client()

        data_parts = df.to_delayed()
        parts = list(map(delayed, data_parts))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)
        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        key_to_part_dict = dict([(str(part.key), part) for part in data_parts])

        print("KEY_TO_PART_DICT: "+ str(key_to_part_dict))

        """
        Build map of workers to parts
        """
        who_has = yield client.who_has(data_parts)
        worker_parts = {}
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])


        """
        Build Numba devicearrays for all coefficient chunks        
        """

        print("CEFFS: " + str(self.coeffs))

        who_has = yield client.who_has()
        print(str(who_has))

        coeff_keys = [coeff[1] for coeff in self.coeffs]

        print("COEFF KEYS: " + str(coeff_keys))

        who_has = yield client.who_has(coeff_keys)

        print("WHO HAS: " + str(who_has))
        worker_map = []
        coeffs_key_to_part_dict = dict([(str(part.key), part) for worker, part in self.coeffs])


        print(str(coeffs_key_to_part_dict))


        for key, workers in who_has.items():
            print(str(key))
            worker = parse_host_port(first(workers))
            worker_map.append((worker, coeffs_key_to_part_dict[key]))

        coeff_devarrays = [(worker, client.submit(to_gpu_array, part, workers=[worker]))
                    for worker, part in worker_map]

        print(str(coeff_devarrays))

        """
        Gather IPC handles and call _predict() on each worker containing data.
        """

        print("WORKER_PARTS: " + str(worker_parts))

        worker_results = {}
        res = []
        for worker, parts in worker_parts.items():

            # Need to fetch coefficient parts on worker
            coeffs_on_worker = filter(lambda x: x[0] == worker, coeff_devarrays)
            coeffs_not_on_worker = filter(lambda x: x[0] != worker, coeff_devarrays)

            ipc_handles = [client.submit(get_ipc_handles, future, workers=[a_worker])
                           for a_worker, future in coeffs_not_on_worker]

            raw_arrays = [arr[1] for arr in coeffs_on_worker]

            # IPC Handles are loaded in separate threads on worker so they can be
            # used to make calls through cython
            worker_results[worker] = (client.submit(_predict_on_worker,
                                                    (parts, ipc_handles, raw_arrays), workers=[worker]))

            res.append(worker_results[worker])

        yield wait(res)

        print("RES: " + str(res))

        # We can safely close our local threads now that the c++ API is holding onto
        # its own resources.
        def close_threads(d):
            ipc_threads, devarrs, result = d
            [t.close() for t in ipc_threads]

        d = [client.submit(close_threads, future) for future in res]
        yield wait(d)

        def join_threads(d):
            ipc_threads, devarrs, result = d
            [t.join() for t in ipc_threads]

        d = [client.submit(join_threads, future) for future in res]
        yield wait(d)

        def get_result(d):
            ipc_threads, devarrs, result = d
            return result

        # Turn resulting cudf future into a dask-cudf and return it. For now, returning the futures
        # pointing to the data.
        ret = [client.submit(get_result, future) for future in res]

        print("RETURN: "+ str(ret))

        yield wait(ret)

        print("RETURN: "+ str(ret))

        return gen.Return(ret)

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
        return default_client().sync(self._do_predict, df)

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

