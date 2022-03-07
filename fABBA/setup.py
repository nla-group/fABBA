from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["extmod/chainApproximation_c.pyx",
                           "extmod/chainApproximation_cm.pyx",
                           "extmod/fabba_agg_c.pyx",
                           "extmod/fabba_agg_cm.pyx",
                           "extmod/inverse_tc.pyx",
                           "separate/aggregation_c.pyx",
                           "separate/aggregation_cm.pyx"]),
)