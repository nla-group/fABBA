import setuptools
from pathlib import Path
from Cython.Build import cythonize
from distutils.command.build import build as build_orig
import numpy

class BuildExtCommand(build_orig):
    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules, language_level=3)
        
with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="fABBA",
    packages=setuptools.find_packages(),
    version="0.4.9",
    setup_requires=["cython>=0.29.4", "numpy>=1.21.1", "scipy>1.6.0"],
    install_requires=["numpy>=1.21.1"],
    ext_modules=cythonize(["fABBA/*.pyx"], include_path=["fABBA"]),
    package_data={"fABBA": ["chainApproximation_c.pyx", "caggregation.pyx", "caggregation_memview.pyx"]},
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="Stefan Guettel, Xinye Chen",
    author_email="stefan.guettel@manchester.ac.uk",
    description="An efficient aggregation based symbolic representation",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/fABBA",
    cmdclass={"build": BuildExtCommand},
    license='BSD 3-Clause'
)
