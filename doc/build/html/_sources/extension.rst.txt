Extensible ABBA
======================================

We also provide other clustering based ABBA methods for extesion, it is easy to use with the support of scikit-learn tools. For users who want to develop their own clustering based ABBA, this tutorial is quite useful, in particular to research comparsion. The user guidance is as follows

.. code:: python

    import numpy as np
    from sklearn.cluster import KMeans
    from fABBA import ABBAbase

    ts = [np.sin(0.05*i) for i in range(1000)]         # original time series
    #  specifies 5 symbols using kmeans clustering
    kmeans = KMeans(n_clusters=5, random_state=0, init='k-means++', verbose=0)     
    abba = ABBAbase(tol=0.1, scl=1, clustering=kmeans)
    string = abba.fit_transform(ts)                    # string representation of the time series
    print(string)                                      # prints BbAaAaAaAaAaAaAaC
    inverse_ts = abba.inverse_transform(string)        # reconstruction



Note `fABBA` software package is not limited to fABBA,  you can directly call ABBA with:

.. code:: python

    from fABBA import ABBA
    abba = ABBA(tol=0.1, scl=1, k=5, verbose=0)
    string = abba.fit_transform(ts)
    print(string)

it will output: 


.. parsed-literal::
    
    Compression: Reduced series of length 1000 to 17 segments. Digitization: Reduced 17 pieces to 5 symbols.
    BbAaAaAaAaAaAaAaC
