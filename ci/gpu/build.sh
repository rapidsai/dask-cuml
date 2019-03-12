#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##############################################
# Dask cuML GPU build and test script for CI #
##############################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcuml and cuML from source
################################################################################

logger "Install cuml"

conda install -c rapidsai -c rapidsai-nightly -c conda-forge -c numba -c rapidsai/label/cuda${CUDA_REL} -c nvidia/label/cuda${CUDA_REL} -c pytorch -c defaults cuml dask distributed cudf dask_cudf

logger "Build Dask cuML..."
cd $WORKSPACE/
python setup.py build_ext --inplace

################################################################################
# TEST - Run GoogleTest and py.tests for libcuml and cuML
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "Python py.test for Dask cuML..."
cd $WORKSPACE
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v
