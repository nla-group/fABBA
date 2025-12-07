.. CLASSIX documentation

Welcome to fABBA's documentation!
===================================
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/yourname/fABBA/blob/main/LICENSE
   :alt: License: MIT

.. image:: https://img.shields.io/pypi/v/fABBA?color=blue
   :target: https://pypi.org/project/fABBA/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/fabba/badge/?version=latest
   :target: https://fabba.readthedocs.io/
   :alt: Documentation Status

fABBA — Fast and Accurate Symbolic Representation for Time Series
==================================================================

**fABBA** (fast ABBA) is a state-of-the-art, highly optimized symbolic aggregate approximation method for univariate and multivariate time series. It achieves extremely high compression ratios (often > 100–1000×) while remaining fully reversible and providing tight error bounds.

The method consists of two core steps:

1. **Lossy piecewise linear compression** (tolerance-driven polygonal chain approximation)
2. **Mean-based clustering of segments -> symbolic representation** (fully automated, no need to pre-specify alphabet size)

Because the resulting representation is symbolic, it naturally leads to:

- Strong noise smoothing
- Drastic dimensionality reduction
- Ultra-fast distance computations (via lookup tables)
- Seamless integration with classic data mining algorithms (motif discovery, anomaly detection, classification, clustering, indexing, etc.)

fABBA significantly outperforms the original ABBA [1]_ in speed (often 10–100× faster) while producing nearly identical or even better symbolic sequences.

.. figure:: images/abba.png
   :width: 720
   :align: center
   :alt: Illustration of the ABBA/fABBA symbolization process

   Visualization of the fABBA transformation process (source: Stefan Güttel, Turing–Manchester presentation, 2021).

Key Advantages of fABBA
-----------------------

=============  ===================================================================
Feature        Description
=============  ===================================================================
Fully automatic      No need to specify number of symbols (α) — purely tolerance-driven
Extremely fast       Sorting + early abandoning + incremental aggregation -> O(n log n) typical
Reversible           Perfect reconstruction via ``inverse_transform``
Multivariate support Unified alphabet across all dimensions (JABBA subclass)
GPU & parallel       Built-in OpenMP and optional CUDA k-means backends
Image compression    Direct 2D block-wise compression with ``image_compress()``
=============  ===================================================================

**Core Methods & Variants**

- ``fABBA.fABBA``           -> Original fast single-series implementation (pure Python + Cython)
- ``fABBA.JABBA``           -> Next-generation engine supporting:
    - Univariate & multivariate series
    - Custom clustering backends (k-means, hierarchical, GPU, etc.)
    - Memory-optimized streaming aggregation
- ``fABBA.image_compress`` / ``image_decompress`` -> Turn any 2D array/image into a short string and back

Applications
------------

fABBA has demonstrated superior performance in numerous domains:

- Time-series classification & clustering (UCR/UEA archives)
- Extreme compression of sensor data (IoT, wearables, finance)
- Motif & discord discovery at massive scale
- Anomaly detection with symbolic distance measures
- Lossy but reconstructible storage of medical signals (ECG, EEG)
- Image and video frame compression via block-wise symbolization

Quick Example
-------------

.. code-block:: python

   from fABBA import fABBA
   import numpy as np
   import matplotlib.pyplot as plt

   ts = np.load("example_series.npy")
   fabba = fABBA(tol=0.1, alpha=0.01, method='agg')
   string, centers = fabba.fit_transform(ts)

   print(f"Original length : {len(ts)}")
   print(f"Compressed to   : {len(string)} symbols  ->  compression ratio {(len(ts)/len(string)):.1f}×")
   print(f"Symbolic string : {string}")

   reconstructed = fabba.inverse_transform(string, centers)

   plt.plot(ts, label="Original")
   plt.plot(reconstructed, "--", label="Reconstructed")
   plt.legend(); plt.show()

References
----------

.. [1] Chen, X. and Güttel, S., 2024. fABBA: A Python library for the fast symbolic approximation of time series.  
   *Journal of Open Source Software* 9(95), 6294.  
   DOI: `10.21105/joss.06294 <https://doi.org/10.21105/joss.06294>`__  
   *(Official software paper describing the fABBA Python package — the foundation of this repository.)*

.. [2] Chen, X., 2024. Joint symbolic aggregate approximation of time series.  
   Preprint arXiv:2401.00109 [cs.DS].  
   https://arxiv.org/abs/2401.00109  
   *(Latest theoretical advance: joint symbolization of multivariate time series with shared alphabet. Fully implemented in the ``JABBA`` class of this package.)*

.. [3] Chen, X. and Güttel, S., 2023. An Efficient Aggregation Method for the Symbolic Representation of Temporal Data.  
   *ACM Transactions on Knowledge Discovery from Data* 17(1), Article 5 (22 pages).  
   DOI: `10.1145/3532622 <https://doi.org/10.1145/3532622>`__  
   *(Core fABBA algorithm paper: introduces sorting-based aggregation, tolerance-driven symbolization, and image compression capabilities.)*

.. [4] Elsworth, S. and Güttel, S., 2020. ABBA: Aggregate Brownian bridge-based aggregation for time series compression and symbolization.  
   *Data Mining and Knowledge Discovery* 34, 1175–1200.  
   DOI: `10.1007/s10618-020-00687-9 <https://doi.org/10.1007/s10618-020-00687-9>`__  
   *(Original ABBA method using k-means clustering on piecewise linear segments.)*

.. [5] Original ABBA implementation (2020):  
   https://github.com/nla-group/ABBA

.. [6] This repository — fABBA + JABBA + latest extensions (2023–2025, actively maintained):  
   https://github.com/nla-group/fABBA  
   *(Includes Multithreading/Cython speedups, GPU-accelerated backends, streaming aggregation, full joint/multivariate support, image utilities, and more. Recommended for all use cases.)*

Getting Started
---------------

.. code-block:: bash

   pip install fABBA          # includes pre-compiled wheels for Linux/macOS/Windows

Full documentation: https://fabba.readthedocs.io

We welcome contributions! Whether it's new clustering backends, performance improvements, or better documentation — feel free to open issues or pull requests.

Enjoy ultra-fast symbolic time-series analysis with fABBA!
    
Guide
-------------

.. toctree::
   :maxdepth: 2
   
   quickstart
   main_comp
   multivariate
   parameters
   extension
   

API Reference
-------------
.. toctree::
   :maxdepth: 2

   api_reference


Others
-------------
.. toctree::
   :maxdepth: 2
   
   license
   contact


Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. image:: images/nla_group.png
    :width: 360
