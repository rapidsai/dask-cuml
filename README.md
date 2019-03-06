# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;Dask cuML - Multi-GPU Machine Learning Algorithms</div>

Dask cuML contains parallel machine learning algorithms that can make use of multiple GPUs on a single host. It is able to play nicely with other projects in the Dask ecosystem, as well as other RAPIDS projects, such as Dask cuDF.

As an example, the following Python snippet loads input from a csv file into a Dask cuDF Dataframe and Performs a NearestNeighbors query:

```python
import dask_cudf
from dask_cuml.neighbors import NearestNeighbors

df = dask_cudf.read_csv("/path/to/csv")

nn = NearestNeighbors(n_neighbors = 10)
nn.fit(df)
nn.kneighbors(df)
```

### Supported Algorithms

- Nearest Neighbors
- Truncated Singular Value Decomposition
- Linear Regression

More ML algorithms are being worked on. 

## Installation

Dask cuML relies on cuML to be installed. Refer to [cuML](https://github.com/rapidsai/cuml) on Github for more information.

#### Conda 

Dask cuML can be installed using the `rapidsai` conda channel:
```bash
conda install -c nvidia -c rapidsai -c conda-forge -c pytorch -c defaults dask_cuml
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
- cuml

Dask cuML can be installed with the following command at the root of the repository:
```bash
python setup.py install
```

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
