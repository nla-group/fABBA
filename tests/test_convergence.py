"""Convergence contracts, separated from non-monotone clustering heuristics."""
import importlib
import unittest
import numpy as np
from numpy.testing import assert_allclose
from fABBA import compress, inverse_compress, fABBA
from fABBA.chainApproximation import compress as python_compress


class ConvergenceTests(unittest.TestCase):
    def test_compression_error_budget(self):
        rng=np.random.default_rng(2026)
        signals=[np.sin(np.linspace(0,12,161)),rng.normal(size=161),
                 np.r_[np.zeros(80),np.ones(81)]]
        for x in signals:
            for tol in [1.,.1,.01,.0001,0.]:
                with self.subTest(tol=tol):
                    p=np.asarray(compress(x,tol=tol))
                    y=np.asarray(inverse_compress(p,x[0]))
                    self.assertEqual(len(x),len(y))
                    assert_allclose(y[[0,-1]],x[[0,-1]],atol=1e-12)
                    # Per-segment SSE <= tol * (length - 1) + roundoff.
                    self.assertTrue(np.all(p[:,2] <= tol*(p[:,0]-1)+1e-12))
                    self.assertLessEqual(np.sum((y-x)**2), tol*(len(x)-1-len(p))+1e-10)
                    if tol == 0: assert_allclose(y,x,atol=1e-12)

    def test_full_pipeline_tight_tolerance_limit(self):
        x=np.sin(np.linspace(0,9,151))+.1*np.cos(np.linspace(0,31,151))
        errors=[]
        for tol,alpha in [(.2,.5),(.001,.05),(0,0)]:
            m=fABBA(tol=tol,alpha=alpha,verbose=0)
            y=np.asarray(m.inverse_transform(m.fit_transform(x),x[0]))
            self.assertEqual(y.shape,x.shape)
            errors.append(np.sqrt(np.mean((y-x)**2)))
        self.assertLess(errors[-1],1e-12)
        self.assertLess(errors[-1],errors[0])
        # No assertion of universal monotonic RMSE: cluster memberships change.

    def test_max_length_exact_difference_limit(self):
        x=np.random.default_rng(4).normal(size=50)
        p=np.asarray(compress(x,tol=100,max_len=1))
        assert_allclose(p[:,0],1)
        assert_allclose(p[:,1],np.diff(x))
        assert_allclose(inverse_compress(p,x[0]),x,atol=1e-12)

    def test_compiled_python_compression_parity(self):
        try: compiled=importlib.import_module('fABBA.extmod.chainApproximation_cm').compress
        except ImportError: self.skipTest('compiled extensions not installed')
        for x in [np.ones(20),np.arange(20.),np.sin(np.linspace(0,8,51))]:
            for tol in [.1,.001,0]:
                with self.subTest(tol=tol):
                    assert_allclose(compiled(x,tol=tol),python_compress(x,tol=tol),atol=1e-12)

if __name__ == '__main__': unittest.main()
