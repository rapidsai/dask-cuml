#
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

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import versioneer
from distutils.sysconfig import get_python_lib

install_requires = [
  'numpy',
  'cuml',
  'cudf',
  'dask_cudf',
  'dask'
]

packages = ["dask_cuml",
            "dask_cuml.tests"]

try:
  numpy_include = numpy.get_include()
except AttributeError:
  numpy_include = numpy.get_numpy_include()
  
setup(name = "dask_cuml",
      description = "Distributed machine on GPUs using Dask",
      version = versioneer.get_version(),
      classifiers = [
        "Intended Audience :: Developers",
        "Programming Landuage :: Python :: 3.5",
      ],
      author = "NVIDIA Corporation",
      setup_requires=[],
      url = "https://github.com/rapidsai/dask_cuml",
      install_requires = install_requires,
      license = "Apache Software License 2.0",
      cmdclass = versioneer.get_cmdClass(),
      packages = packages,
      zip_safe = False
)