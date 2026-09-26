fABBA: symbolic approximation of time series
============================================

.. image:: https://img.shields.io/pypi/v/fABBA?color=2563eb
   :target: https://pypi.org/project/fABBA/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/license-BSD--3--Clause-059669
   :target: https://github.com/nla-group/fABBA/blob/master/LICENSE
   :alt: BSD 3-Clause license

fABBA converts numerical time series into sequences of symbols. Each symbol
represents a learned segment described by its duration and change in value.
The sequence and its codebook can reconstruct a **lossy approximation** of the
original data. Applications include compact representations, exploratory pattern
analysis and features for downstream learning.

Start with :doc:`quickstart` for one series, :doc:`examples` for complete runnable
programs, and :doc:`multivariate` when several series must share a codebook.
The :doc:`parameters` guide explains how accuracy, alphabet size and segment
length interact. :doc:`serialization` explains exactly what to retain for decoding.

What the library does
---------------------

1. Approximate a sampled signal with a continuous polygonal chain.
2. Group the chain's ``[length, increment]`` pieces after feature scaling.
3. Assign a symbol to each group and retain its mean piece as a codebook entry.
4. Decode symbols, round segment lengths to the sample grid, and integrate
   increments from the saved starting value.

Symbol strings are meaningful only with their associated codebooks. Independently
fitted models can assign different meanings to the same character. fABBA does
not make arbitrary symbolic edit distances equivalent to numerical distances.
Compression ratios in samples per symbol are not byte-level storage ratios.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   examples
   parameters
   serialization
   main_comp
   multivariate
   extension

.. toctree::
   :maxdepth: 2
   :caption: Reference and development

   api_reference
   architecture
   testing
   releasing
   license
   contact

Citation
--------

X. Chen and S. Güttel, *fABBA: A Python library for the fast symbolic
approximation of time series*, Journal of Open Source Software 9(95), 6294 (2024).
`doi:10.21105/joss.06294 <https://doi.org/10.21105/joss.06294>`_.
See the repository's ``CITATION.bib`` for additional publications.
