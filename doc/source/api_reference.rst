Public API reference
====================

This reference covers the workflows in the executable user guide. Internal
Cython functions and research-only helpers are not stable public interfaces.

Univariate estimator
--------------------

.. code-block:: python

   fABBA(tol=0.1, alpha=0.5, sorting="2-norm", scl=1, verbose=1,
         partition_rate=None, partition=None, fillna="ffill", max_len=-1,
         return_list=False, n_jobs=1)

* ``fit(series, fillm='bfill', alphabet_set=0) -> self``: learn a new codebook.
* ``fit_transform(...) -> str | ndarray``: learn and return symbols.
* ``compress(series, fillm='bfill')``: return length/increment/error pieces.
* ``digitize(pieces, alphabet_set=0)``: return symbols and a codebook.
* ``inverse_transform(string, start=0, parameters=None) -> list``: decode with
  fitted or supplied parameters. An empty sequence returns ``[start]``.
* ``print_parameters()``: print the learned symbol-to-center mapping.
* ``dump(file=None)`` / ``load(file=None, replace=False)``: legacy trusted-pickle
  codebook persistence. Default path is ``parameters``.

``alphabet_set`` can select a built-in ordering (0 or 1) or supply a list of
unique single-character strings with enough capacity. Codes are Unicode.
``parameters.centers`` has shape ``(n_symbols, 2)`` and columns length/increment.
``parameters.alphabets[i]`` labels center row ``i``. Additional fitted attributes
are ``string_``, ``pieces_``, ``start_`` and ``n_samples_``.

Portable codebook
-----------------

.. autoclass:: fABBA.Model
   :members: to_dict, from_dict

Functional pipeline
-------------------

.. code-block:: python

   compress(series, tol=0.5, max_len=-1, fillm='bfill')
   digitize(pieces, alpha=0.5, sorting='norm', scl=1, alphabet_set=0)
   inverse_digitize(strings, parameters)
   quantize(pieces)
   inverse_compress(pieces, start)
   fillna(series, method='zero')

The standalone ``compress`` and ``digitize`` defaults differ from the estimator's
``tol`` and ``sorting`` defaults. Pass these explicitly when comparing APIs.
``digitize`` returns ``(symbol_array, Model)``. ``inverse_digitize`` returns a
copy of the center rows. ``inverse_compress`` expects integer segment lengths;
use ``quantize`` on fractional learned lengths. See :doc:`main_comp` for the
complete composition.

Other supported workflows
-------------------------

``ABBA(tol=0.1, k=2, scl=1, verbose=1, max_len=-1)`` selects k-means instead of
radius grouping. ``ABBAbase(clustering, ...)`` allows a compatible clustering
object. These estimators expose fit, fit_transform and inverse_transform.

For JABBA, :doc:`multivariate` documents the two-dimensional input convention,
training/transform return types, starting values and reconstruction. Its
``parameters`` type differs from ``Model`` above.

For image-specific helpers, see :doc:`extension`. Dataset loaders are optional
convenience functions; the main tutorials do not rely on external datasets.
