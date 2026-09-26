Runnable examples and toy applications
======================================

Install the package first. Each program below generates its own data, uses a
fixed seed where randomness is involved, runs without network access and prints
numerical results. Run the programs from the repository root::

   python example/toy_models.py
   python example/export_codebook.py
   python example/tolerance_sweep.py
   python example/shared_codebook.py

The CI executes these programs against an installed wheel outside the source
tree. This also checks that examples use packaged public APIs.

Five toy signals
----------------

The constant and linear signals exercise zero-variance grouping. The sine wave
illustrates repeated patterns; the step exercises abrupt changes; the noisy
sine illustrates the tradeoff between detail and compactness. Each row reports
sample count, symbol count, alphabet size and reconstruction RMSE. These counts
are representation sizes, not measured byte compression ratios.

.. literalinclude:: ../../example/toy_models.py
   :language: python

Parameter sweep
---------------

This program reports polygonal and symbolic errors separately. Its final row
checks the zero-tolerance limit. Do not infer a universal monotonic relationship
from this single signal; see :doc:`parameters` and :doc:`testing`.

.. literalinclude:: ../../example/tolerance_sweep.py
   :language: python

Shared-codebook classification
------------------------------

The following toy application distinguishes low and high frequencies using
normalized symbol counts and nearest-neighbor classification. It fits the
codebook on the training data only. The same character then has the same meaning
in both training and test features. Histograms discard ordering, so this is a
workflow example, not an evaluated classifier or anomaly detector.

.. literalinclude:: ../../example/shared_codebook.py
   :language: python

For exporting results to another process, see the complete JSON example in
:doc:`serialization`. Historical notebooks under ``exp/`` are research materials
and may require external datasets; they are separate from this tested gallery.
