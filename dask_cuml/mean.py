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
from cuml import MGMean as cumlMGMean

import core

from tornado import gen
import dask_cudf, cudf

import time

from dask.distributed import get_worker, get_client

from dask import delayed
from collections import defaultdict
from dask.distributed import wait, default_client
import dask.dataframe as dd
import dask.array as da

from toolz import first, assoc
from distributed import Client


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def to_gpu_matrix(df):
    rm = df.as_gpu_matrix(order='F')

    print("GPU: " + str(rm))
    print("CTYPES: "+ str(rm.device_ctypes_pointer))
    return rm


def alloc_dict(gpu_matrix):
    cai = gpu_matrix.__cuda_array_interface__
    return {"ptr": gpu_matrix.device_ctypes_pointer.value,
            "dtype": cai["typestr"],
            "shape": cai["shape"]
    }


def get_ipc_handles(gpu_matrix):
    return gpu_matrix.get_ipc_handle()


def print_it(gpu_matrix):

    print("gpu:  " + str(gpu_matrix))
    print("out: " + str(gpu_matrix[0][0]))
    return gpu_matrix

class MGMean(object):


    def calculate(self, dask_df):
        client = default_client()

        # Keep the futures around so the GPU memory doesn't get
        # deallocated on the workers.
        ipcs, gpu_futures, ipc_futures = client.sync(self._get_mg_info, dask_df)



        # The parts below should be run on a single worker on each unique host




        open_ipcs = [x.open() for x in ipcs]

        dud = [client.submit(print_it, future, workers=worker) for worker, future in gpu_futures]
        wait(dud)

        m = cumlMGMean()
        outs = m.calculate(list(map(alloc_dict, open_ipcs)))

        [x.close() for x in ipcs]



        return outs

    @gen.coroutine
    def _get_mg_info(self, dask_df):

        client = default_client()

        if isinstance(dask_df, dd.DataFrame):
            data_parts = dask_df.to_delayed()
            parts = list(map(delayed, data_parts))
            parts = client.compute(parts)  # Start computation in the background
            yield wait(parts)
            for part in parts:
                if part.status == 'error':
                    yield part  # trigger error locally
        else:
            data_parts = dask_df


        key_to_part_dict = dict([(str(part.key), part) for part in data_parts])

        who_has = yield client.who_has(data_parts)
        worker_map = []

        for key, workers in who_has.items():
            worker_map.append((first(workers), key_to_part_dict[key]))

        gpu_data = [[worker, client.submit(to_gpu_matrix, part, workers=worker)]
                    for worker, part in worker_map]

        ipc_handles = [client.submit(get_ipc_handles, future, workers=worker) for worker, future in gpu_data]

        handles = yield client._gather(ipc_handles)
        handles = [x for x in handles]

        raise gen.Return((handles, gpu_data, ipc_handles))