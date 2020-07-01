from setuptools import setup, find_packages
#from distutils.core import setup
from Cython.Build import cythonize
import numpy as np


setup(
    name='bepler_embedding',
    packages=find_packages(),
    author="Tristan Bepler", 
    version="0.1.0",
    ext_modules = cythonize(['bepler_embedding/metrics.pyx', 'bepler_embedding/alignment.pyx']),
    include_dirs=[np.get_include()]
)
