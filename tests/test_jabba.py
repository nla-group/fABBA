"""Shared-codebook contracts for the documented JABBA workflow."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from fABBA import JABBA
from fABBA.fabba import NotFittedError


class JABBATests(unittest.TestCase):
    def test_training_and_heldout_codebook(self):
        t=np.linspace(0,12,120)
        train=np.array([np.sin(t),np.cos(t)])
        model=JABBA(tol=.01,alpha=.1,verbose=0)
        symbols=model.fit_transform(train.tolist(),n_jobs=1)
        self.assertEqual(len(symbols),2)
        centers=model.parameters.centers.copy()
        alphabet=model.parameters.alphabets.copy()
        encoded,starts=model.transform([np.sin(t+.1).tolist()],n_jobs=1)
        assert_array_equal(model.parameters.centers,centers)
        assert_array_equal(model.parameters.alphabets,alphabet)
        y=model.inverse_transform(encoded,start_set=starts,n_jobs=1)
        self.assertEqual(len(y),1)
        self.assertTrue(np.isfinite(y[0]).all())
        self.assertAlmostEqual(y[0][0],starts[0])

    def test_constant_signals(self):
        model=JABBA(tol=.01,alpha=.1,verbose=0)
        x=np.array([np.ones(20),np.ones(20)*3])
        s=model.fit_transform(x,n_jobs=1)
        assert_allclose(model.inverse_transform(s,n_jobs=1),x)

    def test_missing_values_do_not_mutate(self):
        x=np.array([[np.nan,1.,np.nan,3.,np.nan]])
        before=x.copy()
        model=JABBA(tol=.01,alpha=.1,verbose=0,fillna='bfill')
        model.fit(x,n_jobs=1)
        assert_array_equal(x,before)
        self.assertEqual(model.start_set,[1.])

    def test_univariate_parallel_remainder(self):
        x=np.random.default_rng(8).normal(size=103)
        model=JABBA(tol=.001,alpha=1e-12,max_len=1,verbose=0)
        symbols=model.fit_transform(x,n_jobs=3)
        assert_allclose(model.inverse_transform(symbols,n_jobs=2),x,atol=1e-12)
        encoded,starts=model.transform(x,n_jobs=3)
        assert_allclose(model.inverse_transform(encoded,start_set=starts,n_jobs=2),x,atol=1e-12)
        # Different chunk boundaries expose increments absent from the trained
        # codebook; only sample preservation, not lossless encoding, is promised.
        encoded,starts=model.transform(x,n_jobs=4)
        decoded=model.inverse_transform(encoded,start_set=starts,n_jobs=2)
        self.assertEqual(len(decoded),len(x))
        self.assertTrue(np.isfinite(decoded).all())

    def test_validation(self):
        model=JABBA(tol=.01,alpha=.1,verbose=0)
        with self.assertRaises(NotFittedError): model.transform([[1,2]],n_jobs=1)
        with self.assertRaises(NotFittedError): model.inverse_transform([['A']])
        for value in [[],[1],[[1]],[[1,np.inf]],[[1,1j]]]:
            with self.subTest(value=value),self.assertRaises(ValueError): model.fit(value,n_jobs=1)
        with self.assertRaises(ValueError): model.fit([[1,2]],n_jobs=0)
        symbols=model.fit_transform([[1,2,3]],n_jobs=1)
        with self.assertRaises(ValueError): model.inverse_transform(symbols,start_set=[])

if __name__ == '__main__': unittest.main()
