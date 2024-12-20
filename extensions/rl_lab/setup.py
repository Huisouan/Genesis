#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="rl_lab",
    version="0.0.1",
    packages=find_packages(),
    author="hsh",
    maintainer="hsh",
    maintainer_email="1653996628@qq.com",
    url="",
    license="BSD-3",
    description="genesis RL framework environment",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
    ],
)
