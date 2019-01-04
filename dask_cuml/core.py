from dask.distributed import wait
import random


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_host_dict(workers):
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict


def assign_gpus(client):
    """
    Supports a multi-GPU & multi-Node environment by assigning a single local GPU
    to each worker in the cluster. This is necessary due to Numba's restriction
    that only a single CUDA context (and thus a single device) can be active on a
    thread at a time.

    The GPU assignments are valid as long as the future returned from this function
    is held in scope. This allows any functions that need to allocate GPU data to
    utilize the CUDA context on the same device, otherwise data could be lost.
    """

    workers = list(client.has_what().keys())
    hosts_dict = build_host_dict(workers)

    def get_gpu_info():
        import numba.cuda
        return [x.id for x in numba.cuda.gpus]

    gpu_info = dict([(host,
                      client.submit(get_gpu_info,
                                    workers=[(host, random.sample(hosts_dict[host], 1)[0])]))
                     for host in hosts_dict])
    wait(list(gpu_info.values()))

    # Scatter out a GPU device ID to workers
    f = []
    for host, future in gpu_info.items():
        gpu_ids = future.result()
        ports = random.sample(hosts_dict[host], min(len(gpu_ids), len(hosts_dict[host])))

        f.extend([client.scatter(device_id, workers=[(host, port)]) for device_id, port in zip(gpu_ids, ports)])
    wait(f)

    return f