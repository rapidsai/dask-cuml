#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
import versioneer

install_requires = [
  'numpy'
]

packages = ["dask_cuml",
            "dask_cuml.tests",
            "dask_cuml.neighbors",
            "dask_cuml.linear_model",
            "dask_cuml.randomforest"]

setup(name="dask-cuml",
      description="Distributed ML algorithms on GPUs using Dask",
      version=versioneer.get_version(),
      classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
      ],
      author="NVIDIA Corporation",
      url="https://github.com/rapidsai/dask-cuml",
      install_requires=install_requires,
      license="Apache Software License 2.0",
      cmdclass=versioneer.get_cmdclass(),
      packages=packages,
      zip_safe=False
      )
