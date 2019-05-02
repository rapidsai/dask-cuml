#!/usr/bin/env bash

set -e

echo "Building dask-cuml"

conda build conda/recipes/dask-cuml -c conda-forge -c numba -c rapidsai -c rapidsai-nightly -c nvidia -c pytorch --python=${PYTHON}
