import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy
include_dirs=[numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]

setup(
    ext_modules = cythonize("smc_algo.pyx"),
    include_dirs=[numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
)

