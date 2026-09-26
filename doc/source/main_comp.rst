Use the individual pipeline stages
==================================

The functional API is useful for inspecting compression error independently of
symbolization. It shares digitization and reconstruction logic with ``fABBA``.

.. doctest::

   >>> import numpy as np
   >>> from fABBA import compress, digitize, inverse_digitize, quantize, inverse_compress
   >>> x = np.array([2., 3., 4., 3., 2.])
   >>> pieces = np.asarray(compress(x, tol=0))
   >>> int(pieces[:, 0].sum())
   4
   >>> symbols, codebook = digitize(pieces, alpha=0, sorting="2-norm")
   >>> decoded_pieces = inverse_digitize(symbols, codebook)
   >>> integer_pieces = quantize(decoded_pieces)
   >>> reconstructed = inverse_compress(integer_pieces, start=x[0])
   >>> np.allclose(reconstructed, x)
   True

``compress`` returns rows of ``[length, increment, squared_error]``.
``digitize`` consumes the first two columns and returns a symbol array plus
``Model``. ``inverse_digitize`` maps symbols to independent copies of center
rows. ``quantize`` rounds cumulative lengths to the sample grid without mutating
its input. ``inverse_compress`` integrates the resulting integer-length pieces.

For a fitted signal, group means preserve the total segment duration in exact
arithmetic, and quantization preserves its rounded total. A new or edited symbol
sequence need not have the original sample count. Store the original length and
check reconstruction shape before calculating pointwise errors.

``fABBA.compress`` is the supported validation boundary. Internal Cython modules
operate on typed arrays and are implementation details, not alternate public APIs.
