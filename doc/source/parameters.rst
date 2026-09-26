Parameters, units and error interpretation
==========================================

The two approximation stages have independent controls. Tune them on
representative data and measure the final reconstruction, rather than optimizing
alphabet size alone.

.. list-table:: Core fABBA settings
   :header-rows: 1
   :widths: 18 18 64

   * - Parameter
     - Default
     - Meaning
   * - ``tol``
     - ``0.1``
     - Finite nonnegative compression tolerance, in squared signal units.
   * - ``alpha``
     - ``0.5``
     - Finite nonnegative radius for grouping scaled pieces; dimensionless.
   * - ``scl``
     - ``1``
     - Weight on normalized lengths relative to increments. Zero ignores length
       when grouping, but learned centers still retain lengths.
   * - ``sorting``
     - ``'2-norm'``
     - ``'lexi'``, ``'1-norm'``, ``'2-norm'``, ``'norm'`` or ``'pca'``.
       Sorting changes greedy group assignments and symbol identities.
   * - ``max_len``
     - ``-1``
     - Maximum intervals per polygonal segment. ``-1`` is unlimited;
       a positive integer is required otherwise. ``1`` yields first differences.
   * - ``verbose``
     - ``1``
     - Enable estimator logging. Use zero for quiet examples; configure Python
       logging in the calling application.
   * - ``return_list``
     - ``False``
     - Return the symbol array instead of joining it into a string.

Compression tolerance
---------------------

For a segment spanning :math:`L` sample intervals, the compressor accepts
its endpoint interpolant when

.. math::

   E = \sum_{j=0}^{L}(x_{s+j}-\widehat{x}_{s+j})^2
       \leq \mathrm{tol}(L-1) + \epsilon.

Here :math:`\epsilon` is a floating-point allowance. Length counts intervals,
so :math:`L+1` samples belong to a segment. Adjacent segments share endpoints.
The total polygonal squared error is the sum of segment errors (up to roundoff).
This bound applies **before digitization**. Replacing segments with codebook
means and rounding lengths adds error; ``tol`` is not an end-to-end RMSE bound.

Grouping and scaling
--------------------

The digitizer divides length and increment columns by their standard deviations
and multiplies the length column by ``scl``. A constant column uses a scale of
one, avoiding division by zero. Centers are means in **original units**, not
these normalized coordinates. ``splist`` contains aggregation diagnostics and
has a backend-dependent column layout; do not use it as a portable decoder.

Smaller ``tol`` generally retains more polygonal detail. Smaller ``alpha``
generally preserves more distinct segment types. Neither statement guarantees
monotone full-pipeline RMSE or alphabet size for every input. The algorithms are
greedy, and segment boundaries and memberships can change discontinuously.
``tol=0, alpha=0`` provides a useful exact-limit regression check within
floating-point accuracy, not a promise of bitwise-identical reconstruction.

Choosing settings
-----------------

1. Decide whether amplitude matters. Normalize signals explicitly when it does
   not, and retain the mean and scale for reconstruction in original units.
2. Sweep ``tol`` first with a small ``alpha`` to inspect polygonal error.
3. Adjust ``alpha`` to balance alphabet size and measured reconstruction error.
4. Validate downstream features on held-out data. Fit shared codebooks on the
   training split only.

If a signal is multiplied by :math:`c`, comparable compression tolerance scales
by :math:`c^2`. ``scl`` is not a substitute for signal normalization.
Run ``python example/tolerance_sweep.py`` for a reproducible comparison.

Missing data and shape
----------------------

``fABBA`` accepts a real one-dimensional sequence of at least two samples.
Row and column vectors are accepted for compatibility; other matrices are
rejected rather than silently flattened. Use ``JABBA`` for multiple series.
Infinity is invalid. Inputs are copied before filling NaNs.

Pass ``fillm`` explicitly to ``fit``, ``fit_transform`` or ``compress``; its
default is ``'bfill'``. Supported case-insensitive methods are ``zero``, ``mean``,
``median``, ``ffill`` and ``bfill``. Forward filling uses zero for a leading gap;
backward filling uses zero for a trailing gap. Mean/median filling requires at
least one observed sample. The legacy constructor ``fillna`` setting controls
the partition helper; prefer explicit preprocessing for consistent workflows.

Partitioned compression
-----------------------

For ``fABBA(partition=k, n_jobs=p)``, chunks overlap at their shared endpoints,
so every sample interval, including any remainder, is retained. Partitioning
can change the approximation and should be included in reproducibility settings.
``partition_rate`` derives a chunk count from ``exp(1/rate)`` and the worker
count; an explicit positive ``partition`` is easier to reason about.
``n_jobs=-1`` selects available CPUs. Use ``n_jobs=1`` for predictable small runs.
