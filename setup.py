from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

class CustomBuildExt(build_ext):
    def run(self):
        self.include_dirs.append(numpy.get_include())
        super().run()

ext_modules = [
    Extension("fABBA.extmod.chainApproximation_c",
              ["fABBA/extmod/chainApproximation_c.pyx"]),
    Extension("fABBA.extmod.chainApproximation_cm",
              ["fABBA/extmod/chainApproximation_cm.pyx"]),
    Extension("fABBA.extmod.fabba_agg_c",
              ["fABBA/extmod/fabba_agg_c.pyx"]),
    Extension("fABBA.extmod.inverse_tc",
              ["fABBA/extmod/inverse_tc.pyx"]),
    Extension("fABBA.extmod.fabba_agg_cm",
              ["fABBA/extmod/fabba_agg_cm.pyx"]),
    Extension("fABBA.extmod.fabba_agg_cm_win",
              ["fABBA/extmod/fabba_agg_cm_win.pyx"]),

    Extension("fABBA.separate.aggregation_c",
              ["fABBA/separate/aggregation_c.pyx"]),
    Extension("fABBA.separate.aggregation_cm",
              ["fABBA/separate/aggregation_cm.pyx"]),

    Extension("fABBA.jabba.compmem",
              ["fABBA/jabba/compmem.pyx"]),
    Extension("fABBA.jabba.aggwin",
              ["fABBA/jabba/aggwin.pyx"]),
    Extension("fABBA.jabba.aggmem",
              ["fABBA/jabba/aggmem.pyx"]),
    Extension("fABBA.jabba.inversetc",
              ["fABBA/jabba/inversetc.pyx"])
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
