import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            Extension(
                "circuitlib.signals",
                ["src/circuitlib/signals.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "circuitlib.solver",
                ["src/circuitlib/solver.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ]
    ),
)

