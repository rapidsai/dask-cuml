#!/usr/bin/env bash

set -e

echo "Building dask-cuml"

conda build conda/recipes/dask-cuml -c conda-forge -c numba -c rapidsai/label/cuda${CUDA_REL} -c nvidia/label/cuda${CUDA_REL} -c pytorch -c defaults --python=${PYTHON}
