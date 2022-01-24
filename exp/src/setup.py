from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["compress_c.pyx", "cagg_memview.pyx"]),
)