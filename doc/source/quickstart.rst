Installation and first reconstruction
=====================================

Install a release in a virtual environment::

   python -m venv .venv
   # macOS / Linux:
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install fABBA

This checkout declares Python 3.9 or later and NumPy 1.24 or later. The CI matrix
covers Python 3.9, 3.12, 3.13 and 3.14 on Linux, macOS and Windows. It tests compiled
and Python-only installations. Other versions are not all covered by that matrix.
When a matching wheel is unavailable, pip builds the Cython extensions and needs
a C compiler. A local development installation is ``python -m pip install -e .``.
For a compiler-free source build, set ``FABBA_NO_EXTENSIONS=1`` before installation
(see :doc:`testing`). NumPy and SciPy still need suitable binary distributions.

A complete example
------------------

This example generates its own data and requires no downloads or plot window.

.. doctest::

   >>> import numpy as np
   >>> from fABBA import fABBA
   >>> x = 5 + np.sin(np.linspace(0, 4 * np.pi, 200))
   >>> model = fABBA(tol=0.01, alpha=0.1, verbose=0)
   >>> symbols = model.fit_transform(x)
   >>> reconstructed = np.asarray(model.inverse_transform(symbols, start=x[0]))
   >>> reconstructed.shape == x.shape
   True
   >>> np.isfinite(reconstructed).all().item()
   True
   >>> print("samples:", len(x), "symbols:", len(symbols))  # doctest: +SKIP
   >>> rmse = np.sqrt(np.mean((x - reconstructed) ** 2))

``fit`` returns the estimator. ``fit_transform`` returns a string by default;
with ``return_list=True`` it returns the symbol array. Neither returns a
``(string, centers)`` tuple. Learned entries are available as
``model.parameters.centers`` and ``model.parameters.alphabets``.

``inverse_transform`` defaults to ``start=0``. Supply the original first value
to preserve the vertical position. With NaN filling, use the **filled** first
value (also recorded as ``model.start_`` after fitting).

Inspect the representation
--------------------------

.. doctest::

   >>> model.parameters.centers.shape[1]
   2
   >>> model.pieces_.shape[1]
   3
   >>> int(model.pieces_[:, 0].sum()) == len(x) - 1
   True
   >>> mapping = dict(zip(model.parameters.alphabets, model.parameters.centers))
   >>> first_piece = mapping[symbols[0]]  # [mean length, mean increment]

Use ``model.print_parameters()`` for a readable mapping or :doc:`serialization`
for portable JSON export. Symbol names and exact clustering results can vary
with sorting and numerical backend, so avoid treating a particular character
sequence as a stable identifier across separately trained models.

Optional plotting
-----------------

After the example above, display the signal with::

   import matplotlib.pyplot as plt
   plt.plot(x, label="original")
   plt.plot(reconstructed, "--", label="reconstructed")
   plt.xlabel("sample index")
   plt.ylabel("value")
   plt.legend()
   plt.show()

See :doc:`installation` and :doc:`releasing` for native wheel and NumPy ABI release gates.
