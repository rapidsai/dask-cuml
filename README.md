# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;Dask cuML - Multi-GPU Machine Learning Algorithms</div>

Dask cuML contains parallel machine learning algorithms that can make use of multiple GPUs on a single host. It is able to play nicely with other projects in the Dask ecosystem, as well as other RAPIDS projects, such as Dask cuDF.

## Use

As an example, the following Python snippet loads input from a csv file into a [Dask cuDF](https://github.com/rapidsai/dask-cudf) Dataframe and Performs a NearestNeighbors query in parallel, on multiple GPUs:

```python
# Create a Dask CUDA cluster w/ one worker per device
from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster()

# Read CSV file in parallel across workers
import dask_cudf
df = dask_cudf.read_csv("/path/to/csv")

# Fit a NearestNeighbors model and query it
from dask_cuml.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors = 10)
nn.fit(df)
nn.kneighbors(df)
```

## Dask CUDA Clusters

### Using the LocalCUDACluster()

Clusters of Dask workers can be started in several different ways. One of the simplest methods used in non-CUDA Dask clusters is to use `LocalCluster`. For a CUDA variant of the `LocalCluster` that works well with Dask cuML, check out the `LocalCUDACluster` from the [dask-cuda](https://github.com/rapidsai/dask-cuda) project.

Note: It's important to make sure the `LocalCUDACluster` is instantiated in your code before any CUDA contexts are created (eg. before importing Numba or cudf). Otherwise, it's possible that your workers will all be mapped to the same device. 

### Using the dask-worker command

If you will be starting your workers using the `dask-worker` command, Dask cuML requires that each worker has been started with their own unique `CUDA_VISIBLE_DEVICES` setting. 

For example, a user with a workstation containing 2 devices, would want their workers to be started with the following `CUDA_VISIBLE_DEVICES` settings (one per worker):

```
CUDA_VISIBLE_DEVICES=0,1 dask-worker --nprocs 1 --nthreads 1 scheduler_host:8786
```
```
CUDA_VISIBLE_DEVICES=1,0 dask-worker --nprocs 1 --nthreads 1 scheduler_host:8786
```

This enables each worker to map the device memory of their local cuDFs to separate devices.

Note: If starting Dask workers using `dask-worker`,  `--nprocs 1` must be used. 

## Supported Algorithms

- Nearest Neighbors
- Linear Regression

More ML algorithms are being worked on. 

## Installation

Dask cuML relies on cuML to be installed. Refer to [cuML](https://github.com/rapidsai/cuml) on Github for more information.

#### Conda 

Dask cuML can be installed using the `rapidsai` conda channel (if you have CUDA 9.2 installed, change the `cudatoolkit=10.0`
dependency to `cudatoolkit=9.2` instead):

```bash
conda install -c nvidia -c rapidsai -c conda-forge -c defaults dask-cuml cudatoolkit=10.0
```

#### Pip

Dask cuML can also be installed using pip. 
```bash
pip install dask-cuml
```

#### Build/Install from Source

Dask cuML depends on:
- dask
- dask_cudf
- dask_cuda
- cuml

Dask cuML can be installed with the following command at the root of the repository:
```bash
python setup.py install
```

Tests can be verified using Pytest:
```bash
py.test dask_cuml/test
```

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
