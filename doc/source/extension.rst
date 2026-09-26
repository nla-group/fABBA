Variants and extension points
=============================

``fABBA`` uses sorted greedy aggregation and a radius ``alpha``. ``ABBA`` uses
k-means with a requested cluster count ``k``. ``ABBAbase`` accepts a clustering
object exposing ``fit_predict``. A cluster count must be feasible for the
available compressed pieces. These classes share the univariate representation
of length/increment centers and a symbol alphabet.

.. doctest::

   >>> import numpy as np
   >>> from fABBA import ABBA
   >>> x = np.array([0., 1., 0., 1., 0.])
   >>> model = ABBA(tol=0.001, k=2, max_len=1, verbose=0)
   >>> symbols = model.fit_transform(x)
   >>> np.allclose(model.inverse_transform(symbols, start=x[0]), x)
   True

The ``jabba`` package contains JABBA, QABBA and XABBA variants. QABBA adds
quantization choices; XABBA uses a different approximation strategy. Their
interfaces and accuracy properties should not be inferred from the fABBA
convergence tests. Optional accelerated clustering dependencies are only relevant
to the variants that use them. Start with the tested univariate or shared-codebook
examples before adopting a research extension.

Image helpers
-------------

``image_compress(model, image, adjust=True)`` flattens an array into a signal and
records image shape and optional normalization on the estimator.
``image_decompress(model, symbols)`` reconstructs, rounds and casts to uint8.
This conversion is intended for 8-bit image workflows, not arbitrary floating
point arrays. It does not provide a general-purpose lossless image codec.
For a floating point array, reshape explicitly and use the numerical APIs.
