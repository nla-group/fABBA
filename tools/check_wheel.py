"""Run outside the source tree to certify an installed compiled wheel."""
import importlib
import importlib.machinery
from pathlib import Path
import runpy
import sys
import unittest
import numpy as np


def main():
    import fABBA
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, package {fABBA.__file__}")
    # The numerical suite also calls every one of the 12 native extension modules.
    for module_name, attributes in {
        "fABBA.fabba": ("compress", "aggregate_fc", "aggregate_fabba"),
        "fABBA.jabba.jabba": ("compress", "aggregate", "inv_transform"),
    }.items():
        module = importlib.import_module(module_name)
        for attribute in attributes:
            function = getattr(module, attribute)
            backend = importlib.import_module(function.__module__)
            if not any(backend.__file__.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
                raise RuntimeError(f"{module_name}.{attribute} silently fell back to Python")
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit("Compiled wheel verification failed or skipped a required test")
    for name in ("toy_models", "export_codebook", "tolerance_sweep", "shared_codebook"):
        runpy.run_path(str(Path("example") / f"{name}.py"), run_name="__main__")


if __name__ == "__main__":
    main()
