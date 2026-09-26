Architecture and maintenance boundaries
=======================================

Public flow
-----------

.. code-block:: text

   raw series
      | _validation.py + fillna: shape, finite values, copied input
      v
   fabba.py: estimator orchestration and fitted attributes
      | compression dispatch
      +--> extmod/*.pyx: compiled polygonal approximation
      +--> chainApproximation.py: Python fallback
      v
   digitization.py: shared scaling, aggregation, centers, symbols
      | fabba_agg / separate aggregation backends
      v
   Model: centers + alphabets + aggregation diagnostics
      | inverse_t.py: validated decoding and cumulative length rounding
      v
   reconstructed numerical series

``fABBA.fabba.fABBA`` owns configuration and fitted state. Its digitization
method delegates to the functional implementation so both APIs use the same
normalization, grouping and model class. ``Model`` remains importable from its
historical module to support existing pickle references. ``inverse_t.py`` is
the shared public univariate reconstruction path; legacy compiled inverse
kernels remain internal to other code paths.

``fABBA.jabba`` contains a separate family of joint and quantized algorithms.
It is not presented as a subclass of the univariate estimator. Cross-variant
model interchange is not an API guarantee.

Repository layout
-----------------

* ``fABBA/``: installable Python modules and Cython sources.
* ``tests/``: network-free unittest contracts and numerical regression tests.
* ``example/``: runnable examples; the four gallery scripts run in CI.
* ``doc/source/``: Sphinx documentation; gallery code is included from scripts.
* ``exp/`` and ``paper/``: historical experiments and publication material.
* ``build/`` and ``dist/``: generated artifacts, not authoritative sources.

The checkout historically contains tracked generated files. New build output
is ignored; Cython now generates C under ``build/cython`` rather than rewriting
tracked source files. Packaging excludes cached bytecode and prebuilt shared
libraries from package data. Compiled extensions are built for the target
interpreter instead of shipping another machine's binaries.

Compatibility and deliberate behavior changes
---------------------------------------------

The core constructors and fit/decode entry points are retained. Validation now
rejects infinity, invalid tolerances, empty/single-sample signals and ambiguous
matrices. Zero ``tol`` and ``alpha`` are accepted for exact-limit checks.
Unknown symbols raise ``ValueError`` instead of being disguised as not-fitted
errors. Filling NaNs and quantizing pieces no longer mutate caller arrays.
Custom alphabets must contain unique single characters and work at exact
capacity. Partitioned fABBA now retains all intervals, which can change results
from older behavior that discarded boundaries and trailing samples.

JABBA univariate splitting also retains remainder samples. Its chunks are
independent series with saved starts, unlike fABBA's overlapping polygonal
partitions. Changing JABBA chunk boundaries can change the representation.

New attributes after fitting are ``pieces_`` (three-column polygonal pieces),
``start_`` and ``n_samples_``. New JSON helpers are ``Model.to_dict`` and
``Model.from_dict``. Reconstruction uses cumulative rounding, so numerical
outputs can differ from legacy rounding behavior. See :doc:`testing` for
validation scope and limits.
