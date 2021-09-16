import setuptools
from Cython.Build import cythonize
import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="fABBA",
    packages=setuptools.find_packages(),
    version="0.6.5",
    setup_requires=["cython>=0.29.4", "numpy>=1.20.0", "scipy>1.6.0"],
    install_requires=["numpy>=1.20.0", "tqdm", "pandas", "matplotlib"],
    ext_modules=cythonize(["fABBA/*.pyx"], include_path=["fABBA"]),
    package_data={"fABBA": ["chainApproximation_c.pyx", "aggregation_c.pyx", "aggregation_memview.pyx", "fabba_agg_memview.pyx"]},
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="Stefan Guettel, Xinye Chen",
    author_email="stefan.guettel@manchester.ac.uk",
    description="An efficient aggregation based symbolic representation",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/fABBA",
    license='BSD 3-Clause'
)
