Mult-channel symbolization
======================================


**JABBA** is a fast, parallel, and fully multivariate-aware implementation of the **fABBA** (fast Adaptive Brownian Bridge-based Approximation) symbolic aggregation method for time series.

It extends the original ABBA with:

- Native support for multivariate and high-dimensional arrays (images, video frames, sensor arrays)
- Automatic shape preservation and restoration
- Parallel compression & digitization via multiprocessing
- Three digitization backends: adaptive aggregation (original ABBA), K-means, and GPU-accelerated K-means
- Out-of-sample transformation
- Auto-digitization (no need to tune ``alpha``)

Perfect for: motif discovery, compression, clustering, classification, anomaly detection on **real-world multivariate** data.

Core Idea
---------

1. Compress each time series into piecewise linear segments (length, increment)
2. Digitize all pieces across all series/channels into shared symbols
3. Reconstruct using starting points + symbolic sequence

Symbols are consistent across all variables -> enables cross-channel pattern mining.


Quick Start – One-Liner
-----------------------

.. code-block:: python

    from fABBA import JABBA
    import numpy as np

    # 50 multivariate time series, 6 channels, 500 timesteps each
    X = np.random.randn(50, 6, 500)

    jabba = JABBA(tol=0.05, verbose=1)
    symbols = jabba.fit_transform(X)           # List[List[str]] — one sequence per series
    X_reconstructed = jabba.inverse_transform(symbols)

    print(f"Reconstruction error: {np.linalg.norm(X - X_reconstructed):.4f}")


Full Usage Examples
===================


1. Multiple Univariate or Multivariate Time Series (Most Common)
----------------------------------------------------------------

.. code-block:: python

    from fABBA import JABBA
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulate 20 independent univariate series
    np.random.seed(0)
    data = np.cumsum(np.random.randn(20, 800), axis=1)  # random walks

    jabba = JABBA(tol=0.1, init='agg', verbose=1)  # auto-digitization
    symbols = jabba.fit_transform(data)
    recon = jabba.inverse_transform(symbols)

    # Plot first 3 series
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(data[i], label='Original', alpha=0.8)
        plt.plot(recon[i], '--', label='JABBA reconstruction')
        plt.title(f'Series {i} -> compressed to {len(symbols[i])} symbols')
        plt.legend()
    plt.tight_layout()
    plt.show()


2. True Multivariate Time Series (Shared Symbols Across Channels)
------------------------------------------------------------------

.. code-block:: python

    # 10 samples × 12 channels × 1000 timesteps (e.g., EEG, accelerometers)
    mts = np.random.randn(10, 12, 1000)

    jabba = JABBA(tol=0.02, scl=2.0, verbose=1)
    symbols = jabba.fit_transform(mts)        # 10 symbolic sequences (one per sample)
    recon = jabba.inverse_transform(symbols)  # shape: (10, 12, 1000)

    error = np.mean([np.linalg.norm(mts[i] - recon[i]) for i in range(10)])
    print(f"Avg reconstruction error per sample: {error:.4f}")
    print(f"Number of unique symbols: {len(jabba.parameters.alphabets)}")


3. High-Dimensional Arrays (Video, Spectrograms, Images over Time)
------------------------------------------------------------------

.. code-block:: python

    # 8 video clips: 30 frames × 112 × 112 × 3
    video = np.random.rand(8, 30, 112, 112, 3)

    jabba = JABBA(tol=0.1, verbose=1)
    symbols = jabba.fit_transform(video)                    # treats as 8 × (30, 112*112*3) series
    flat_recon = jabba.inverse_transform(symbols)           # (8, 30*112*112*3)
    video_recon = jabba.recast_shape(flat_recon)          # -> (8, 30, 112, 112, 3)

    print("Original shape:", video.shape)
    print("Restored shape :", video_recon.shape)
    print("Max abs error   :", np.max(np.abs(video - video_recon)))


Important: ``recast_shape`` only works if input was a NumPy array (not list/tensor).


4. Out-of-Sample (Test Set) Symbolization
-----------------------------------------

.. code-block:: python

    X_train = np.random.randn(100, 5, 200)
    X_test  = np.random.randn(30, 5, 200)

    jabba = JABBA(tol=0.05).fit(X_train)                    # learn vocabulary
    symbols_test, starts = jabba.transform(X_test)          # use same symbols!
    X_test_recon = jabba.inverse_transform(symbols_test, starts)

    print(f"Test set reconstructed with {len(jabba.parameters.alphabets)} shared symbols")


5. Fixed vs Adaptive Vocabulary
-------------------------------

.. code-block:: python

    data = np.random.randn(50, 4, 1000)

    # Adaptive (recommended): let JABBA decide how many symbols
    adaptive = JABBA(tol=0.03, init='agg', verbose=0)
    adaptive.fit_transform(data)
    print("Adaptive -> symbols:", len(adaptive.parameters.alphabets))

    # Fixed vocabulary (faster, reproducible)
    fixed = JABBA(tol=0.03, init='kmeans', k=80, verbose=0)
    fixed.fit_transform(data)
    print("Fixed k=80 -> symbols:", len(fixed.parameters.alphabets))


6. GPU-Accelerated Digitization (Large Datasets)
------------------------------------------------

.. code-block:: python

    huge_data = np.random.randn(1000, 20, 2000)  # 40 million points

    jabba = JABBA(tol=0.05, init='gpu-kmeans', k=200, verbose=1)
    symbols = jabba.fit_transform(huge_data, n_jobs=16)  # blazing fast


Parameter Guide
===============

================================  ===============================================  =====================
Parameter                         Meaning                                          Default
================================  ===============================================  =====================
``tol``                           Compression tolerance (smaller = more pieces)    0.2
``init``                          Digitization method                                      ``'agg'``
                                  • ``'agg'`` -> original adaptive merging (best quality)
                                  • ``'kmeans'`` -> fast CPU K-means
                                  • ``'gpu-kmeans'`` -> FAISS GPU (very fast)
``k``                             Fixed number of symbols (used only if not 'agg') —
``alpha``                         Merging threshold in aggregation (if given)      ``None`` (auto)
``eta``                           Controls auto-alpha strength (1–5 typical)       3
``scl``                           Weight on length vs increment (higher = respect length more) 1.0
``verbose``                       Print progress                                           1
``n_jobs``                        Parallel workers (-1 = all cores)                        -1
``last_dim``                      Keep last dim as time? (True for (samples,time,feat)) True
================================  ===============================================  =====================


When to Use Which Mode?
-----------------------

+----------------------------------+--------------------------------------------+
| Goal                             | Recommended Settings                       |
+==================================+============================================+
| Best reconstruction quality      | ``init='agg'``, small ``tol``              |
+----------------------------------+--------------------------------------------+
| Fast & reproducible              | ``init='kmeans'``, fixed ``k``             |
+----------------------------------+--------------------------------------------+
| >1M time points                  | ``init='gpu-kmeans'`` + high ``n_jobs``    |
+----------------------------------+--------------------------------------------+
| Motif discovery across channels  | ``init='agg'`` or ``kmeans`` (shared vocab)|
+----------------------------------+--------------------------------------------+


Tips from the Source Code
-------------------------

- JABBA automatically standardizes data unless you set ``adjust=False`` in ``general_compress``
- For peak-shift robustness -> increase ``scl`` (e.g. ``scl=3–5``)
- For very noisy data -> slightly increase ``tol``
- Use ``jabba.parameters.centers`` and ``.alphabets`` to inspect learned prototypes
- ``symbols`` is a list of lists of strings — perfect input for ``pysax``, ``matrix profile``, or ``LSTM+embedding``


You're now ready to symbolize anything from ECG to satellite imagery.

Happy compressing!