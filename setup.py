import subprocess
import sys

from setuptools import Extension, setup

setup_args = dict(
    ext_modules=[
        Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=["megatron/core/datasets/helpers.cpp"],
            language="c++",
            extra_compile_args=(
                subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"])
                .decode("utf-8")
                .strip()
                .split()
            )
            + ["-O3", "-Wall", "-std=c++17"],
            optional=True,
        )
    ]
)
setup(**setup_args)
