Testing, builds and numerical contracts
=======================================

Local checks
------------

From a fresh virtual environment in the repository root::

   python -m pip install -e ".[test,docs]"
   python -m unittest discover -s tests -v
   python runtest.py
   python -m build
   python -m sphinx -W --keep-going -b html doc/source doc/_build/html
   python -m sphinx -W --keep-going -b doctest doc/source doc/_build/doctest

``runtest.py`` is a compatibility entry point for the same unittest suite;
there is no need to run both test commands routinely. ``python -m pytest`` also
collects this suite and is limited to ``tests/`` to avoid historical experiments.

A source build without Cython extensions is explicitly selectable::

   # macOS / Linux
   FABBA_NO_EXTENSIONS=1 python -m pip install .
   # Windows PowerShell
   $env:FABBA_NO_EXTENSIONS = "1"
   python -m pip install .

Use fresh environments and clean build output when switching backends. Merely
setting this variable at runtime does not unload already installed extensions.
The CI builds wheels from source distributions and tests them outside the
checkout, including an assertion that the requested backend is loaded.

What convergence means here
---------------------------

The suite tests per-segment squared-error budgets and the total polygonal error
bound, preservation of endpoints and sample count, and reconstruction in the
zero-tolerance limit. ``max_len=1`` is checked against exact first differences.
The full symbolic pipeline is tested as both tolerances approach zero on a
fixed deterministic signal. Tests do **not** assert universal monotone RMSE,
cluster counts or a particular symbol string.

Other regression contracts include malformed inputs, NaN boundary handling,
non-mutation, zero-variance grouping, custom alphabets, decoding errors,
functional/estimator equivalence, JSON/pickle roundtrips, partition remainders
in fABBA and JABBA, and avoiding plotting initialization during package import.
Compiled/Python compression parity runs when extensions are available; its
explicit skip is expected in the Python-only matrix jobs.

Automation and limits
---------------------

The Tests workflow covers Linux/macOS/Windows with Python 3.9, 3.12, 3.13 and 3.14,
in compiled and Python-only modes. A separate job builds documentation with
warnings treated as errors and executes doctests. The Docker workflow builds
an installable notebook image; the publication workflow builds the paper PDF.
These definitions are only exercised remotely after changes are pushed.

The most recent inspected failing GitHub Pages run (28744936821) built its
artifact successfully, then failed in deployment with ``Deployment failed,
try again later`` while synchronizing files. This was a hosting-stage failure,
not a Python test or Sphinx error. A local source change cannot verify that
GitHub's deployment service has recovered. See ``MAINTENANCE.md`` for the
local verification record and remaining environment-specific checks.

Research variants, GPU backends, external datasets and performance claims need
separate validation before use in a specific application. Passing the core
suite is not a proof of correctness for every variant or every input scale.

See :doc:`installation` and :doc:`releasing` for native wheel and NumPy ABI release gates.
