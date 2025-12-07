.. _api-reference:

API Reference
=============

fABBA
-----

The ``fABBA`` class provides the original FABBA algorithm for symbolic representation of univariate time series.

.. autoclass:: fABBA.fABBA
   :members:
   :undoc-members:
   :show-inheritance:

ABBAbase
--------

Base class shared by fABBA and JABBA.

.. autoclass:: fABBA.ABBAbase
   :members:
   :private-members:

JABBA
-----

``JABBA`` is an extended and highly optimized version that supports:
- Univariate and multivariate time series
- Multiple clustering backends (including GPU-accelerated)
- Memory-efficient and parallel aggregation

.. autoclass:: fABBA.JABBA
   :members:
   :undoc-members:
   :show-inheritance:

Core Transformation Methods
===========================

These functions/methods are the building blocks used internally and can also be used directly.

compress
--------

Perform piecewise linear aggregation (tolerance-based chain approximation).

.. automethod:: fABBA.chainApproximation.compress

inverse_compress
----------------

Reconstruct time series from compressed piecewise aggregates.

.. autofunction:: fABBA.inverse_compress

digitize
--------

Convert piecewise linear segments into symbolic representation (SAX-like).

.. automethod:: fABBA.digitization.digitize

inverse_digitize
----------------

Reconstruct approximate time series from symbolic string and centers.

.. autofunction:: fABBA.inverse_digitize

Image Compression Utilities
===========================

Convenient APIs for compressing 2D arrays/images using fABBA.

image_compress
--------------

Compress a 2D image/array into a symbolic string using block-wise fABBA.

.. autofunction:: fABBA.image_compress

image_decompress
----------------

Decompress a symbolic string back into an image/array.

.. autofunction:: fABBA.image_decompress

Dataset Loading Utilities
==========================

.. autofunction:: fABBA.load_datasets.load_ucr_dataset

.. autofunction:: fABBA.load_datasets.load_uea_dataset

Other Utilities
===============

.. autofunction:: fABBA.fabba_agg.fabba_agg

.. autofunction:: fABBA.fabba_agg.inverse_fabba_agg