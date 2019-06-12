#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##############################################
# Dask cuML GPU build and test script for CI #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export BRANCH_VERSION=0.8.*

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf

logger "conda install -c nvidia/label/cuda$CUDA_REL -c rapidsai-nightly/label/cuda$CUDA_REL -c rapidsai/label/cuda$CUDA_REL -c conda-forge \
    cuml=$BRANCH_VERSION dask distributed cudf=$BRANCH_VERSION dask-cudf=$BRANCH_VERSION dask-cuda=$BRANCH_VERSION"

conda install -c nvidia/label/cuda$CUDA_REL -c rapidsai-nightly/label/cuda$CUDA_REL -c rapidsai/label/cuda$CUDA_REL -c conda-forge \
    cuml=$BRANCH_VERSION dask distributed cudf=$BRANCH_VERSION dask-cudf=$BRANCH_VERSION dask-cuda=$BRANCH_VERSION

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build dask-cuml from source
################################################################################

logger "Build Dask cuML..."
cd $WORKSPACE/
python setup.py build_ext --inplace

################################################################################
# TEST - Run GoogleTest and py.tests for libcuml and cuML
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Python py.test for Dask cuML..."
cd $WORKSPACE
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v
