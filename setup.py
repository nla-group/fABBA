import logging
import setuptools
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import numpy

with open("README.rst", 'r') as f:
    long_description = f.read()

logging.basicConfig()
log = logging.getLogger(__file__)
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)

setup_args = {'name':"fABBA",
        'packages':setuptools.find_packages(),
        'version':"0.9.3",
        'setup_requires':["numpy"],
        'cmdclass': {'build_ext': build_ext},
        'install_requires':["cython", "numpy", "scipy>=1.2.1", "requests", "pandas"],
        'package_data':{"fABBA": [
                                 "extmod/chainApproximation_c.pyx", 
                                 "extmod/chainApproximation_cm.pyx",
                                 "separate/aggregation_c.pyx", 
                                 "separate/aggregation_cm.pyx",
                                 "separate/aggregation.py", 
                                 "extmod/fabba_agg_c.pyx",
                                 "extmod/fabba_agg_cm.pyx",
                                 "extmod/inverse_tc.pyx"],
                       },
        'include_dirs':[numpy.get_include()],
        'long_description':long_description,
        'author':"Xinye Chen, Stefan Güttel",
        'author_email':"xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
        'classifiers':["Intended Audience :: Science/Research",
                    "Intended Audience :: Developers",
                    "Programming Language :: Python",
                    "Topic :: Software Development",
                    "Topic :: Scientific/Engineering",
                    "Operating System :: Microsoft :: Windows",
                    "Operating System :: Unix",
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3.6",
                    "Programming Language :: Python :: 3.7",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10"
                    ],
        'description':"An efficient aggregation method for the symbolic representation of temporal data",
        'long_description_content_type':'text/x-rst',
        'url':"https://github.com/nla-group/fABBA",
        'license':'BSD 3-Clause'
    }

try:
    setuptools.setup(
        ext_modules=cythonize(["fABBA/extmod/*.pyx", "fABBA/separate/*.pyx"], include_path=["fABBA"]), **setup_args
    )
except ext_errors as ext_reason:
    log.warn(ext_reason)
    log.warn("The C extension could not be compiled.")
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']
    setup(**setup_args)