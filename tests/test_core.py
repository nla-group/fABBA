"""Public API contracts and regression tests; no datasets or network required."""
import contextlib
import io
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from fABBA import fABBA, ABBA, Model, compress, inverse_compress, digitize, inverse_digitize, quantize, fillna
from fABBA.fabba import NotFittedError


class CoreTests(unittest.TestCase):
    def test_import_does_not_initialize_plotting(self):
        subprocess.run([sys.executable, "-c",
                        "import sys, fABBA; assert 'matplotlib.pyplot' not in sys.modules"],
                       check=True, capture_output=True, text=True)

    def test_roundtrip_toy_series(self):
        rng = np.random.default_rng(42)
        signals = [np.ones(50), np.arange(31.), np.sin(np.linspace(0, 8, 100)),
                   np.r_[np.zeros(12), np.ones(15), np.zeros(8)], rng.normal(size=80), [1., 2.]]
        for signal in signals:
            for sorting in ['lexi', '1-norm', '2-norm', 'norm', 'pca']:
                with self.subTest(signal=np.shape(signal), sorting=sorting):
                    model = fABBA(tol=0, alpha=0, sorting=sorting, verbose=0)
                    symbols = model.fit_transform(signal)
                    decoded = model.inverse_transform(symbols, start=signal[0])
                    assert_allclose(decoded, signal, atol=1e-12)
                    self.assertTrue(np.isfinite(model.parameters.centers).all())
                    self.assertEqual(model.n_samples_, len(signal))

    def test_missing_values_copy_and_boundary(self):
        x = np.array([np.nan, 1, np.nan, 3, np.nan])
        expected = {'bfill': [1,1,3,3,0], 'ffill': [0,1,1,3,3],
                    'zero': [0,1,0,3,0], 'Mean': [2,1,2,3,2], 'Median': [2,1,2,3,2]}
        for method, values in expected.items():
            with self.subTest(method=method):
                assert_allclose(fillna(x, method), values)
                model = fABBA(tol=0, alpha=0, verbose=0)
                symbols = model.fit_transform(x.tolist(), fillm=method)
                assert_allclose(model.inverse_transform(symbols, start=values[0]), values)
        assert_array_equal(np.isnan(x), [True,False,True,False,True])
        with self.assertRaises(ValueError): fillna([np.nan, np.nan], 'mean')
        with self.assertRaises(ValueError): fillna([1,2], 'typo')

    def test_input_validation(self):
        for x in [[], [1], [[1,2],[3,4]], [1,np.inf], [1,complex(1,2)], 3]:
            with self.subTest(x=x), self.assertRaises((ValueError, TypeError)):
                fABBA(verbose=0).fit_transform(x)
        for value in [-1, np.nan, np.inf, True, 'small']:
            for parameter in ['tol', 'alpha', 'scl']:
                with self.subTest(parameter=parameter,value=value), self.assertRaises((TypeError, ValueError)):
                    fABBA(**{parameter:value})
        for value in [0,-2,1.5,np.inf,True]:
            with self.subTest(max_len=value), self.assertRaises((ValueError, TypeError)):
                compress([1,2,3],max_len=value)
        for pieces in [[], [[0,1]], [[1,np.inf]], [1,2]]:
            with self.subTest(pieces=pieces), self.assertRaises(ValueError): digitize(pieces)

    def test_vector_shapes_and_noncontiguous_input(self):
        x = np.arange(20.)[::2]
        expected = compress(x)
        for value in [x.reshape(1,-1), x.reshape(-1,1), x.tolist()]:
            assert_allclose(compress(value), expected)

    def test_decode_errors_and_empty_sequence(self):
        model = fABBA(verbose=0)
        with self.assertRaises(NotFittedError): model.inverse_transform('A')
        model.fit([1,2,3])
        with self.assertRaisesRegex(ValueError, 'symbol'): model.inverse_transform('?')
        with self.assertRaises(ValueError): model.inverse_transform('A', start=np.nan)
        self.assertEqual(model.inverse_transform('', start=3), [3.])

    def test_functional_class_parity(self):
        x = np.sin(np.linspace(0, 12, 200))
        model = fABBA(tol=.01, alpha=.1, sorting='2-norm', verbose=0)
        symbols = model.fit_transform(x)
        pieces = compress(x, tol=.01)
        s, parameters = digitize(pieces, alpha=.1, sorting='2-norm')
        self.assertEqual(symbols, ''.join(s))
        assert_allclose(parameters.centers, model.parameters.centers)
        y = inverse_compress(quantize(inverse_digitize(s, parameters)), x[0])
        assert_allclose(y, model.inverse_transform(symbols,x[0]))

    def test_custom_alphabet_exact_capacity(self):
        model = fABBA(tol=0, alpha=0, max_len=1, verbose=0)
        symbols = model.fit_transform([0,1,0], alphabet_set=['↑','↓'])
        self.assertEqual(set(symbols), {'↑','↓'})
        assert_allclose(model.inverse_transform(symbols), [0,1,0])
        for alphabet in [['A'], ['A','A'], ['up','down']]:
            with self.assertRaises(ValueError): model.fit([0,1,0],alphabet_set=alphabet)

    def test_json_export_and_independent_decoder(self):
        model = fABBA(tol=.01,alpha=.1,verbose=0).fit(np.sin(np.linspace(0,8,100)))
        data = json.loads(json.dumps(model.parameters.to_dict(), allow_nan=False))
        restored = Model.from_dict(data)
        fresh = fABBA(verbose=0)
        assert_allclose(fresh.inverse_transform(model.string_, model.start_, restored),
                        model.inverse_transform(model.string_, model.start_))
        restored.centers[0,1] += 1
        self.assertNotEqual(restored.centers[0,1],model.parameters.centers[0,1])
        for bad in [{}, dict(data,schema_version=2),dict(data,alphabets=['A','A']),dict(data,centers=[[0,1]])]:
            with self.assertRaises(ValueError): Model.from_dict(bad)

    def test_pickle_and_print_parameters(self):
        model=fABBA(verbose=0).fit([0,1,2,3])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'codebook.pkl'
            model.dump(path)
            other=fABBA(verbose=0)
            other.load(path, replace=True)
            assert_allclose(other.inverse_transform(model.string_),[0,1,2,3])
        with contextlib.redirect_stdout(io.StringIO()) as output: model.print_parameters()
        self.assertIn('length=',output.getvalue())

    def test_partition_preserves_all_intervals(self):
        x=np.sin(np.linspace(0,10,103))
        for partition in [1,2,7,200]:
            with self.subTest(partition=partition):
                model=fABBA(tol=0,alpha=0,partition=partition,n_jobs=2,verbose=0)
                symbols=model.fit_transform(x)
                self.assertEqual(model.pieces_[:,0].sum(),len(x)-1)
                assert_allclose(model.inverse_transform(symbols,x[0]),x,atol=1e-12)

    def test_quantize_conserves_duration_without_mutation(self):
        pieces=np.array([[1.5,2],[1.5,-2],[2.2,1],[1.8,0]])
        before=pieces.copy()
        q=quantize(pieces)
        assert_array_equal(pieces,before)
        self.assertTrue(np.all(q[:,0]>=1))
        self.assertEqual(q[:,0].sum(),round(pieces[:,0].sum()))
        self.assertEqual(len(inverse_compress(q,0)),8)
        assert_array_equal(quantize([[1.5, 0], [1., 0]])[:, 0], [2., 1.])

    def test_abba_equal_lengths(self):
        model=ABBA(tol=.001,k=2,max_len=1,verbose=0)
        x=np.array([0,1,0,1,0.])
        assert_allclose(model.inverse_transform(model.fit_transform(x)),x)

if __name__ == '__main__': unittest.main()
