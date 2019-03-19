#!/usr/bin/env bash

set -e

echo "Building dask-cuml"

conda build conda/recipes/dask-cuml -c conda-forge -c numba -c rapidsai/label/cuda${CUDA} -c nvidia/label/cuda${CUDA} -c pytorch -c defaults --python=${PYTHON}
