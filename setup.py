import logging
import setuptools
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import Extension

try:
    from Cython.Distutils import build_ext
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools.command.build_ext import build_ext
    
with open("README.rst", 'r') as f:
    long_description = f.read()

logging.basicConfig()
log = logging.getLogger(__file__)
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)
        
setup_args = {'name':"fABBA",
        'packages':setuptools.find_packages(),
        'version':"1.2.4",
        'cmdclass': {'build_ext': CustomBuildExtCommand},
        'install_requires':["numpy>=1.3.0", "scipy>=0.7.0", 
                            "requests", "pandas", 
                            "scikit-learn", 
                            "joblib>=1.1.1",
                            "matplotlib"],
        'packages':{"fABBA", "fABBA.extmod", "fABBA.separate", "fABBA.jabba"},
        'package_data':{"fABBA": ["jabba/data/*.npy", "jabba/data/*.arff"]},
        # 'include_dirs':[numpy.get_include()],
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
                    "Programming Language :: Python :: 3"
                    ],
        'description':"An efficient aggregation method for the symbolic representation of temporal data",
        'long_description_content_type':'text/x-rst',
        'url':"https://github.com/nla-group/fABBA",
        'license':'BSD 3-Clause'
    }


chainApproximation_c = Extension('fABBA.extmod.chainApproximation_c',
                        sources=['fABBA/extmod/chainApproximation_c.pyx'])

chainApproximation_cm = Extension('fABBA.extmod.chainApproximation_cm',
                        sources=['fABBA/extmod/chainApproximation_cm.pyx'])

fabba_agg_c = Extension('fABBA.extmod.fabba_agg_c',
                        sources=['fABBA/extmod/fabba_agg_c.pyx'])

inverse_tc = Extension('fABBA.extmod.inverse_tc',
                        sources=['fABBA/extmod/inverse_tc.pyx'])


fabba_agg_cm = Extension('fABBA.extmod.fabba_agg_cm',
                        sources=['fABBA/extmod/fabba_agg_cm.pyx'])

fabba_agg_cm_win = Extension('fABBA.extmod.fabba_agg_cm_win',
                        sources=['fABBA/extmod/fabba_agg_cm_win.pyx'])

aggregation_c = Extension('fABBA.separate.aggregation_c',
                        sources=['fABBA/separate/aggregation_c.pyx'])

aggregation_cm = Extension('fABBA.separate.aggregation_cm',
                        sources=['fABBA/separate/aggregation_cm.pyx'])


compmem_j = Extension('fABBA.jabba.compmem',
                        sources=['fABBA/jabba/compmem.pyx'])

aggwin_j = Extension('fABBA.jabba.aggwin',
                        sources=['fABBA/jabba/aggwin.pyx'])

aggmem_j = Extension('fABBA.jabba.aggmem',
                        sources=['fABBA/jabba/aggmem.pyx'])

inversetc_j = Extension('fABBA.jabba.inversetc',
                        sources=['fABBA/jabba/inversetc.pyx'])


try:
    from Cython.Build import cythonize
    setuptools.setup(
        setup_requires=["cython", "numpy>=1.17.3"],
        # ext_modules=cythonize(["fABBA/extmod/*.pyx", 
        #                        "fABBA/separate/*.pyx"], 
        #                      include_path=["fABBA/fABBA"]), 
        **setup_args,
        ext_modules=[chainApproximation_c,
                     chainApproximation_cm,
                     fabba_agg_c,
                     fabba_agg_cm,
                     fabba_agg_cm_win,
                     inverse_tc,
                     aggregation_c,
                     aggregation_cm,
                     compmem_j,
                     aggwin_j,
                     aggmem_j,
                     inversetc_j
                    ],
    )
    
except ext_errors as ext_reason:
    log.warn(ext_reason)
    log.warn("The C extension could not be compiled.")
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']
    setuptools.setup(setup_requires=["numpy>=1.17.3"], **setup_args)
    
