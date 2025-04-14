from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "QuantCoreStats",
        ["StatsFunctions.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="QuantCoreStats",
    ext_modules=ext_modules,
    version="0.0.1",
    description="A test module using PyBind11",
)