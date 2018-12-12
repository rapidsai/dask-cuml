set -e

echo "Building dask_cuml"
conda build conda-recipes -c rapidsai -c numba -c conda-forge -c defaults --python $PYTHON