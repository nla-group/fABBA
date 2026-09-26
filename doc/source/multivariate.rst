Multiple series with a shared JABBA codebook
============================================

Use ``JABBA`` when several signals must share symbol meanings, especially for
classification features or held-out inference. For the simplest workflow, pass
a finite float array with shape ``(n_series, n_samples)`` and use ``n_jobs=1``.
Each row is a series. Explicitly reshape multichannel data to this convention
when channel orientation would otherwise be ambiguous.

.. doctest::

   >>> import numpy as np
   >>> from fABBA import JABBA
   >>> t = np.linspace(0, 4 * np.pi, 120)
   >>> train = np.asarray([np.sin(t), np.cos(t)])
   >>> model = JABBA(tol=0.01, alpha=0.1, verbose=0, random_state=42)
   >>> training_symbols = model.fit_transform(train, n_jobs=1)
   >>> test = np.asarray([np.sin(t + 0.1)])
   >>> test_symbols, starts = model.transform(test, n_jobs=1)
   >>> reconstructed = model.inverse_transform(test_symbols, start_set=starts, n_jobs=1)
   >>> len(test_symbols) == len(test)
   True
   >>> np.isfinite(np.asarray(reconstructed)).all().item()
   True

``fit_transform`` returns symbol sequences; ``transform`` returns
``(symbol_sequences, start_set)``. Preserve those new starting values when
reconstructing held-out data. A codebook trained on different durations can
produce a different decoded sample count, so inspect lengths rather than
assuming pointwise alignment for all transformed signals.

Use explicit ``alpha`` for a controlled initial experiment. ``alpha=None``
enables JABBA's automatic digitization selection. Its settings, model state,
parallel partition behavior and quantized variants are distinct from the
univariate ``fABBA`` interface. JABBA does not inherit from ``ABBAbase``.

The complete classification program in :doc:`examples` shows how to build
features without fitting a second codebook on test data. Independently fitting
one ``fABBA`` model per row would not provide comparable symbol identities.
