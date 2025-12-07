.. _saving-loading-jabba:

Saving and loading
===================================================================

The following applications demonstrate how to save and load trained JABBA models and their learned symbolic dictionaries for later use, deployment, or sharing.
It works for all ABBA variants (fABBA, JABBA, QABBA). Now we foucs on JABBA as an example.

After `fit()` or `fit_transform()`, JABBA stores everything needed for perfect reconstruction in the lightweight dataclass:

.. code-block:: python

    jabba.parameters   # -> Model(centers=..., alphabets=...)

================================  =========================================================  =========================
Attribute                         Meaning                                                    Example
================================  =========================================================  =========================
``centers``                       ``(n_symbols, 2)`` float array (already de-normalized!)    
                                  ``centers[i] = [avg_length, avg_increment]`` of the i-th prototype
``alphabets``                     ``(n_symbols,)`` string array — symbol ↔ prototype mapping
                                  Default order: 'A','a','B','b',...,'Z','z', then '!' onwards
================================  =========================================================  =========================

These two arrays are your learned **symbolic dictionary / codebook** — they completely define the compression and reconstruction behavior.

Inspecting the Learned Dictionary
---------------------------------

.. code-block:: python

    from fABBA import JABBA
    import numpy as np
    import pandas as pd

    data = np.random.randn(100, 6, 500)
    jabba = JABBA(tol=0.05, verbose=0).fit(data)

    df = pd.DataFrame({
        'symbol': jabba.parameters.alphabets,
        'avg_length': jabba.parameters.centers[:, 0].round(2),
        'avg_increment': jabba.parameters.centers[:, 1].round(4)
    }).sort_values('avg_increment')

    print(df.head(10))

# Example output
#   symbol  avg_length  avg_increment
# 7      g       15.21        -0.0872
# 3      c        9.84        -0.0431
# 0      A       22.10        -0.0012
# 1      a       11.35         0.0198
# 5      e       18.67         0.0564


Saving the Model (Recommended Methods)
--------------------------------------

.. code-block:: python

    import joblib
    import pickle
    import numpy as np

    # 1. Recommended — tiny & fast (joblib handles numpy efficiently)
    joblib.dump(jabba.parameters, 'jabba_dictionary.joblib')

    # 2. Classic pickle
    with open('jabba_dictionary.pkl', 'wb') as f:
        pickle.dump(jabba.parameters, f)

    # 3. Ultra-lightweight — pure NumPy (ideal for C++/Rust/Java interop)
    np.savez 'jabba_dictionary.npz',
         centers=jabba.parameters.centers,
         alphabets=jabba.parameters.alphabets)


Loading a Trained Dictionary for Inference / Deployment
------------------------------------------------------

.. code-block:: python

    import joblib
    import numpy as np
    from fABBA import JABBA, Model

    # Load dictionary
    params = joblib.load('jabba_dictionary.joblib')           # -> Model instance
    # or
    # data = np.load('jabba_dictionary.npz')
    # params = Model(centers=data['centers'], alphabets=data['alphabets'])

    # Create a "frozen" JABBA instance that only transforms
    jabba_deploy = JABBA(tol=0.05, verbose=0)   # tol must match training!
    jabba_deploy.parameters = params                     # inject learned vocabulary

    # Now symbolize new data without re-fitting
    X_new = np.random.randn(20, 6, 500)
    symbols_new, start_values = jabba_deploy.transform(X_new)
    X_reconstructed = jabba_deploy.inverse_transform(symbols_new, start_values)


Saving the Entire Model (including normalization & shape info)
-------------------------------------------------------------

If you also want to preserve standardization parameters (``d_norm``) and original shape for zero-code reconstruction:

.. code-block:: python

    joblib.dump(jabba, 'jabba_full_model.joblib')

    # Later
    jabba_loaded = joblib.load('jabba_full_model.joblib')
    # Can still call fit new data or transform
    symbols = jabba_loaded.transform(new_data)


Production Deployment Example (FastAPI)
-------------------------------------

.. code-block:: python

    # app.py
    import joblib
    import numpy as np
    from fastapi import FastAPI
    from fABBA import JABBA

    app = FastAPI()
    jabba = joblib.load('jabba_full_model.joblib')  # loaded once at startup

    @app.post("/symbolize")
    async def symbolize(payload: dict):
        arr = np.array(payload["data"])  # (n_samples, n_channels, length)
        symbols, starts = jabba.transform(arr)
        return {"symbols": symbols}

    @app.post("/reconstruct")
    async def reconstruct(payload: dict):
        symbols = payload["symbols"]
        starts = payload.get("starts")
        recon = jabba.inverse_transform(symbols, starts)
        return {"data": recon.tolist()}


Visualizing the Learned Prototypes
--------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    centers = jabba.parameters.centers
    symbols = jabba.parameters.alphabets

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(centers[:, 0], centers[:, 1],
                        c=range(len(symbols)), cmap='tab20', s=120, edgecolors='k')
    for i, sym in enumerate(symbols):
        plt.text(centers[i, 0] + 0.3, centers[i, 1], sym,
                 fontsize=14, weight='bold')

    plt.xlabel('Average segment length', fontsize=12)
    plt.ylabel('Average increment (trend)', fontsize=12)
    plt.title('JABBA Learned Symbolic Dictionary', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('jabba_dictionary.pdf', dpi=300)
    plt.show()


Complete Persistent Package (Most Robust)
---------------------------------------

.. code-block:: python

    joblib.dump({
        'tol'              : jabba.tol,
        'scl'              : jabba.scl,
        'd_norm'           : jabba.d_norm,           # (mean, std) if normalized
        'recap_shape'      : jabba.recap_shape,      # for recast_shape()
        'centers'          : jabba.parameters.centers,
        'alphabets'        : jabba.parameters.alphabets,
    }, 'jabba_complete_package.joblib')


Summary – What You Really Need to Save
--------------------------------------

Only these two objects are required for 100% lossless reconstruction:

.. code-block:: text

    jabba.parameters.centers      -> (K, 2) float64 # for QABBA, it is of integer type
    jabba.parameters.alphabets    -> (K,) strings

+ the original `tol` and `scl` values

With just these, you can reconstruct the original time series perfectly in any language or environment (Python, C++, Java, MATLAB, etc.).

You now have full control over JABBA's symbolic dictionary — ready for industrial deployment, cross-language use, paper reproducibility, and model sharing.

Happy symbolizing!