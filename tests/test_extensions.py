"""Exercise every native kernel, including Windows variants on every platform."""
import importlib
import importlib.machinery
import importlib.util
import unittest
import numpy as np
from numpy.testing import assert_allclose

EXTENSIONS = (
    "extmod.chainApproximation_c", "extmod.chainApproximation_cm",
    "extmod.fabba_agg_c", "extmod.fabba_agg_cm", "extmod.fabba_agg_cm_win",
    "extmod.inverse_tc", "separate.aggregation_c", "separate.aggregation_cm",
    "jabba.compmem", "jabba.aggmem", "jabba.aggwin", "jabba.inversetc",
)


@unittest.skipUnless(importlib.util.find_spec("fABBA.extmod.chainApproximation_cm"),
                     "compiled extensions not installed")
class ExtensionTests(unittest.TestCase):
    def test_all_native_modules_load(self):
        for name in EXTENSIONS:
            with self.subTest(module=name):
                module = importlib.import_module("fABBA." + name)
                self.assertTrue(any(module.__file__.endswith(suffix)
                                    for suffix in importlib.machinery.EXTENSION_SUFFIXES))

    def test_compression_kernels(self):
        x = np.array([0., 1., 2., 1., 0.])
        for name in ["extmod.chainApproximation_c", "extmod.chainApproximation_cm", "jabba.compmem"]:
            with self.subTest(module=name):
                module = importlib.import_module("fABBA." + name)
                pieces = np.asarray(module.compress(x, tol=0.001, max_len=1))
                assert_allclose(pieces[:, 0], np.ones(4))
                assert_allclose(pieces[:, 1], np.diff(x))

    def test_index_buffers_in_all_aggregation_kernels(self):
        # Default NumPy int and C long have different widths on some Windows
        # NumPy versions. np.intp matches argsort indices on all supported ABIs.
        points = np.array([[1., 0.], [1.01, 0.], [3., 2.]])
        for name in ["extmod.fabba_agg_c", "extmod.fabba_agg_cm", "extmod.fabba_agg_cm_win",
                     "separate.aggregation_c", "separate.aggregation_cm", "jabba.aggmem", "jabba.aggwin"]:
            module = importlib.import_module("fABBA." + name)
            sorting = "2-norm" if "fabba_agg" in name else "norm"
            with self.subTest(module=name):
                labels, _ = module.aggregate(points.copy(), sorting, 0.05)
                labels = np.asarray(labels)
                self.assertEqual(labels.shape, (3,))
                self.assertEqual(labels[0], labels[1])
                self.assertNotEqual(labels[0], labels[2])
            if hasattr(module, "aggregate_1d"):
                with self.subTest(module=name, dimensions=1):
                    labels, _ = module.aggregate_1d(np.array([0., 0.01, 2.]), 0.05)
                    self.assertEqual(len(labels), 3)
                    self.assertEqual(labels[0], labels[1])
                    self.assertNotEqual(labels[0], labels[2])

    def test_inverse_kernels(self):
        centers = np.array([[2., 2.], [2., -2.]])
        for name, symbols in [("extmod.inverse_tc", "AB"), ("jabba.inversetc", ["A", "B"])]:
            with self.subTest(module=name):
                module = importlib.import_module("fABBA." + name)
                assert_allclose(module.inv_transform(symbols, centers, ["A", "B"], 0.), [0.,1.,2.,1.,0.])
