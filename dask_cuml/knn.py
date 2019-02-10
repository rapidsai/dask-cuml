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

from cuml import KNN as cumlKNN
from cuml import device_of_ptr

import logging

import random

import itertools


from tornado import gen
import dask_cudf, cudf

import logging

import os
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

def to_gpu_matrix(df):

    try:
        gpu_matrix = df.as_gpu_matrix(order='F')

        dev = device_of_ptr(gpu_matrix)

        # Return canonical device id as string
        return os.environ["CUDA_VISIBLE_DEVICES"].split()[dev], gpu_matrix

    except Exception as e:
        import traceback
        logging.error("Error in to_gpu_matrix(dev=" + str(dev) + "): " + str(e))
        traceback.print_exc()
        pass

def build_alloc_info(data):
    dev, gpu_matrix = data
    return gpu_matrix.__cuda_array_interface__


def get_ipc_handle(data):
    dev, gpu_matrix = data

    try:
        in_handle = gpu_matrix.get_ipc_handle()
        return dev, in_handle
    except Exception as e:
        import traceback
        logging.error("Error in get_ipc_handles(dev=" + str(dev) + "): " + str(e))
        traceback.print_exc()
        pass


def extract_model(data):
    ipcs, rawarrs, m = data
    return m


# Run on a single worker on each unique host
def _fit(data, params):


    print("DATA: " + str(data))
    print("PARAMS: " + str(params))

    ipcs, raw_arrs = data

    # Separate threads to hold pointers to separate devices
    # The order in which we pass the list of IPCs to the thread matters and the goal is
    # to maximize reuse while minimizing the number of threads. We want to limit the
    # number of threads to O(len(devices)) and want to avoid having if be O(len(ipcs))
    # at all costs!
    device_handle_map = defaultdict(list)
    [device_handle_map[dev].append(ipc) for dev, ipc in ipcs]

    print("device_handle_map=" + str(device_handle_map))

    open_ipcs = [new_ipc_thread(ipcs, dev) for dev, ipcs in device_handle_map.items()]

    alloc_info = list(itertools.chain(*[t.info() for t in open_ipcs]))
    alloc_info.extend([build_alloc_info(t) for t in raw_arrs])

    print("alloc_info=" + str(alloc_info))

    m = cumlKNN(should_downcast = params["should_downcast"])
    m.fit_mg(params["D"], alloc_info)

    return open_ipcs, raw_arrs, m


def _kneighbors(X, m, all_ranks, params):
    from mpi4py import MPI

    print("params: "+ str(params))
    print("all_ranks: "  + str(all_ranks))

    return m.query_mn(X, params["k"], all_ranks)


class KNN(object):
    """
    Data-parallel Multi-Node Multi-GPU kNN Model.

    Data is spread across Dask workers using Dask cuDF. On each unique host, a single worker is chosen to creates
    a series of kNN indices, one for each chunk of the Dask input, across devices on that host. Each unique hostname
    is assigned a monotonically increasing identifier, which is used as a multiplier for the resulting kNN indices
    across hosts so that the global index matrix, returned from queries, will reflect the global order.
    """

    def __init__(self, should_downcast = False):
        self.sub_models = []
        self.host_masters = []
        self.should_downcast = should_downcast

    def fit(self, ddf):
        """
        Fits a multi-node multi-gpu knn model, each node using their own index structure underneath.
        :param futures:
        :return:
        """
        client = default_client()

        # Keep the futures around so the GPU memory doesn't get
        # deallocated on the workers.
        gpu_futures, cols = client.sync(self._get_mg_info, ddf)

        print("gpu_futures=" + str(gpu_futures))

        print("D=" + str(cols))

        # Choose a random worker on each unique host to run cuml's kNN.fit() function
        # on all the cuDFs living on that host
        master_hosts = [(host, random.sample(ports, 1)[0])
                        for host, ports in self._build_host_dict(gpu_futures, client).items()]

        self.host_masters = [(worker, client.submit(get_ranks, ident, workers=[worker]).result())
                             for worker, ident in zip(master_hosts, range(len(master_hosts)))]



        print("HOST MASTERS: " + str(self.host_masters))


        f = []
        for host, port in master_hosts:

            gpu_futures_for_host = list(filter(lambda d: d[0][0] == host, gpu_futures))

            print("gpu_futures_for_host=" + str(gpu_futures_for_host))

            exec_node = (host, port)

            # build ipc handles
            gpu_data_excl_worker = list(filter(lambda d: d[0] != exec_node, gpu_futures_for_host))
            gpu_data_incl_worker = list(filter(lambda d: d[0] == exec_node, gpu_futures_for_host))

            ipc_handles = [client.submit(get_ipc_handle, future, workers=[worker])
                           for worker, future in gpu_data_excl_worker]

            raw_arrays = [future for worker, future in gpu_data_incl_worker]

            print("raw_arrays=" + str(raw_arrays))

            f.append((exec_node, client.submit(_fit, (ipc_handles, raw_arrays),
                                   {"D": cols, "ranks": self.host_masters, "should_downcast":self.should_downcast},
                                   workers=[exec_node])))

        wait(f)

        self.terminate_ipcs(client, f)

        # The model on each unique host is held for futures queries
        self.sub_models = dict([(worker, client.submit(extract_model, future, workers = [worker]))
                                for worker, future in f])

    @staticmethod
    def terminate_ipcs(client, f):

        # We can safely close our local threads now that the c++ API is holding onto
        # its own resources.

        def close_threads(d):
            ipc_threads, rawarrays, m = d
            [t.close() for t in ipc_threads]

        d = [client.submit(close_threads, future) for worker, future in f]
        wait(d)

        def join_threads(d):
            ipc_threads, rawarrays, m = d
            [t.join() for t in ipc_threads]

        d = [client.submit(join_threads, future) for worker, future in f]

        wait(d)

    def kneighbors(self, X, k):
        """
        Queries the multi-node multi-gpu knn model by propagating the query cudf to each unique host.
        Eventually, this will support dask_cudf inputs but for now it supports a single cudf.

        1. Push X to the master worker on each unique host (cloudpickle serializer should
           allow X to be smart for minimizing copies as much as possible)
        2. Run the kNN query on each master worker

        :param input:
            A cudf to distribute across the workers to run the kNN in parallel.
            NOTE: This is a single cudf for the first iteration and will become a
            dask_cudf in future iterations.
        :param k:
            The number of nearest neighbors to query for each input vector.
        :return:
            dists and indices of the k-nearest neighbors to the input vectors
        """

        client = default_client()

        X_replicated = dict([(worker, client.scatter(X, workers=[worker]))
                            for worker, rank in self.host_masters])

        ranks = [r for h, r in self.host_masters]

        results = [(worker, client.submit(_kneighbors, X_part, self.sub_models[worker], ranks,
                                          {"k": k},workers=[worker]))
                   for worker, X_part in X_replicated.items()]

        # first rank listed in host_masters provides the actual output
        return list(filter(lambda x: x[0] == self.host_masters[0][0], results))[0]

    def get(self, indices):
        """
        Returns the vectors from the knn index for a list of indices.
        :param indices:
        :return:
        """
        pass

    @staticmethod
    def _build_host_dict(gpu_futures, client):

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
    def _get_mg_info(self, ddf):

        client = default_client()

        if isinstance(ddf, dd.DataFrame):
            cols = len(ddf.columns)
            parts = ddf.to_delayed()
            parts = client.compute(parts)
            yield wait(parts)
        else:
            raise Exception("Input should be a Dask DataFrame")

        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        print("KEY TO PART DICT: " + str(key_to_part_dict))

        who_has = yield client.who_has(parts)

        print("WHO HAS: " + str(who_has))

        worker_map = []
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            worker_map.append((worker, key_to_part_dict[key]))

        print("WORKER_MAP: " + str(worker_map))

        gpu_data = [(worker, client.submit(to_gpu_matrix, part, workers=[worker]))
                    for worker, part in worker_map]

        yield wait(gpu_data)

        raise gen.Return((gpu_data, cols))
