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

from cuml import MGKNN as cumlMGKNN
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

from toolz import first, assoc

import numba.cuda

def to_gpu_matrix(inp):
    dev, df = inp
    select_device(dev)

    try:
        gpu_matrix = df.as_gpu_matrix(order='F')
        shape = df.shape[1]
        dtype = gpu_matrix.dtype
        z = np.zeros(shape, dtype=dtype)
        series = cudf.Series(z)
        gpu_array = series._column._data.to_gpu_array()
        return (dev, gpu_matrix, series, gpu_array)
    except Exception as e:
        import traceback
        logging.error("Error in to_gpu_matrix(dev=" + str(dev) + "): " + str(e))
        traceback.print_exc()
        pass


def build_alloc_info(data):
    dev, gpu_matrix, series, gpu_array = data
    return [gpu_matrix.__cuda_array_interface__, gpu_array.__cuda_array_interface__]


def get_ipc_handles(data):
    dev, gpu_matrix, series, gpu_array = data

    select_device(dev)
    try:
        logging.warn("Building in_handle on " + str(dev))
        in_handle = gpu_matrix.get_ipc_handle()

        logging.warn("Building out_handle on " + str(dev))
        out_handle = gpu_array.get_ipc_handle()
        return (dev, in_handle, out_handle)
    except Exception as e:
        import traceback
        logging.error("Error in get_ipc_handles(dev=" + str(dev) + "): " + str(e))
        traceback.print_exc()
        pass


def to_pandas(data):
    dev, gpu_matrix, series, gpu_array = data
    return series.to_pandas()


# Run on a single worker on each unique host
def _fit(data):
    ipcs, raw_arrs = data

    # Get device from local gpu_futures (should be the same)
    select_device(raw_arrs[0][0])

    def new_ipc_thread(dev, ipcs):
        t = IPCThread(ipcs, dev)
        t.start()
        return t

    # Separate threads to hold pointers to separate devices
    open_ipcs = [new_ipc_thread(dev, [inp, outp]) for dev, inp, outp in ipcs]

    m = cumlMGKNN()

    alloc_info = [t.info() for t in open_ipcs]
    alloc_info.extend([build_alloc_info(t) for t in raw_arrs])

    logging.debug("calc_mean_device: " + str(numba.cuda.get_current_device()))
    m.calculate(alloc_info)

    logging.debug("end calc_mean_device: " + str(numba.cuda.get_current_device()))
    return open_ipcs, raw_arrs



class MGKNN(object):
    """
    Data-parallel Multi-Node Multi-GPU kNN Model.

    Data is spread across Dask workers. On each node, a single worker creates a series of indices, one for each
    chunk of the Dask input. Each unique hostname is assigned a monotonically increasing identifier, which is
    used is added to the resulting indices so that queries can be reconstructed properly.
    """

    def fit(self, futures):
        """
        Fits a multi-node multi-gpu knn model, each node using their own index structure underneath.
        :param futures:
        :return:
        """
        client = default_client()

        # Keep the futures around so the GPU memory doesn't get
        # deallocated on the workers.
        gpu_futures = client.sync(self._get_mg_info, futures)

        f = []
        for host, ports in self._build_host_dict(gpu_futures, client).items():

            # Choose a random worker on each node to become the
            # execution node.
            exec_node = (host, random.sample(ports, 1)[0])

            # build ipc handles
            gpu_data_excl_worker = filter(lambda d: d[0] != exec_node, gpu_futures)
            gpu_data_incl_worker = filter(lambda d: d[0] == exec_node, gpu_futures)

            ipc_handles = [client.submit(get_ipc_handles, future, workers=[worker])
                           for worker, future in gpu_data_excl_worker]

            raw_arrays = [future for worker, future in gpu_data_incl_worker]

            f.append(client.submit(_fit, (ipc_handles, raw_arrays), workers=[exec_node]))

        wait(f)

        # We can safely close our local threads now that the c++ API is holding onto
        # its own resources.

        def close_threads(d):
            ipc_threads, rawarrays = d
            [t.close() for t in ipc_threads]

        d = [client.submit(close_threads, future) for future in f]
        wait(d)

        def join_threads(d):
            ipc_threads, rawarrays = d
            [t.join() for t in ipc_threads]

        d = [client.submit(join_threads, future) for future in f]

        wait(d)

        # The model on each node is encapsulated within this model.
        self.sub_models = f

    def

    def kneighbors(self, input):
        """
        Queries the multi-node multi-gpu knn model by propagating the query to each unique host.

        :param input:
        :return:
            dists and indices of the k-nearest neighbors to the input vectors
        """

        # Retrieve k neighbors from 1 worker on each node and have 1 rank fetch results.

        # Need to wrap the MNMG call for each of the self.sub_models to search.

        #


        pass

    def get(self, indices):
        """
        Returns the vectors from the knn index for a list of indices.
        :param indices:
        :return:
        """
        pass

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


    @gen.coroutine
    def _get_mg_info(self, futures):

        client = default_client()

        if isinstance(futures, dd.DataFrame):
            data_parts = futures.to_delayed()
            parts = list(map(delayed, data_parts))
            parts = client.compute(parts)  # Start computation in the background
            yield wait(parts)
            for part in parts:
                if part.status == 'error':
                    yield part  # trigger error locally
        else:
            data_parts = futures

        key_to_part_dict = dict([(str(part.key), part) for part in data_parts])

        who_has = yield client.who_has(data_parts)
        worker_map = []
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            worker_map.append((worker, key_to_part_dict[key]))

        gpu_data = [(worker, client.submit(to_gpu_matrix, part, workers=[worker]))
                    for worker, part in worker_map]

        yield wait(gpu_data)

        raise gen.Return(gpu_data)