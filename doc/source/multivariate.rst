Multivariate time series symbolization
======================================


Here we domonstrate how to use ``fABBA`` to symbolize multivariate (same applies to multiple univariate time series) with consistent symbols. After downloading the `UEA time series dataset <https://www.timeseriesclassification.com/>`_ in corresponding folder, you can run JABBA following the example below:


.. code:: python

    import os
    from scipy.io import arff
    from fABBA import JABBA
    import matplotlib.pyplot as plt
    import numpy as np

    _dir = 'data/UEA2018' # your data file location

    def preprocess(data):
        time_series = list()
        for ii in data[0]:
            database = list()
            for i in ii[0]:
                database.append(list(i))
            time_series.append(database)
        return np.nan_to_num(np.array(time_series))

    filename = 'BasicMotions'
    num= 10
    data = arff.loadarff(os.path.join(_dir, os.path.join(filename, filename+'_TRAIN.arff')))
    multivariate_ts = preprocess(data)

    mts =((multivariate_ts[num].T - multivariate_ts[num].T.mean(axis=0)) /multivariate_ts[num].T.std(axis=0)).T

    jabba1 = JABBA(tol=0.0002, verbose=1)
    symbols_series = jabba1.fit_transform(mts)
    reconstruction = jabba1.inverse_transform(symbols_series)
    
    jabba2 = JABBA(tol=0.0002, init='k-means', k=jabba1.parameters.centers.shape[0], verbose=0)
    symbols_series = jabba2.fit_transform(mts)
    reconstruction_ABBA = jabba2.inverse_transform(symbols_series)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 5))
    
    for i in range(2):
        for j in range(3):
            ax[i,j].plot(mts[i*3 + j], c='yellowgreen', linewidth=5,label='time series')
            ax[i,j].plot(reconstruction_ABBA[i*3 + j], c='blue', linewidth=5, alpha=0.3,label='reconstruction - J-ABBA')
            ax[i,j].plot(reconstruction[i*3 + j], c='purple', linewidth=5, alpha=0.3,label='reconstruction - J-fABBA')
    
            ax[i,j].set_title('dimension '+str(i*3 + j))
            ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
    
    plt.legend(loc='lower right', bbox_to_anchor=[-0.5, -0.5], framealpha=0.45)
    plt.show()


.. image:: images/jabba/all_BasicMotions56.png
    :width: 720


fABBA enable symbolic approximation of multidimentioanl array. Users simply can recontruct the symbols into original shape via ``recast_shape`` . 

.. code:: python
    
    from fABBA import JABBA
    import numpy as np
    mts = np.random.randn(10, 20, 30) # 6000 time series values
    
    jabba = JABBA(tol=0.01, alpha=0.01, verbose=1)
    symbols = jabba.fit_transform(mts)
    reconst = jabba.inverse_transform(symbols) # convert into array
    reconst_same_shape = jabba.recast_shape(reconst) # recast into original shape
    np.linalg.norm((mts - reconst_same_shape).reshape(-1, np.prod(mts.shape[1:])), 'fro')

If one would like to ensure the ``recast_shape`` for shape reconstruction, the input to ``fit_transform`` must be numpy.ndarray.


Regarding the transformation of out-of-sample data, use

.. code:: python

    mts = np.random.randn(20, 20, 30) # new 6000 time series values
    symbols_trans, start_set = jabba.transform(mts) # Perform transform with fitted model
    reconst = jabba.inverse_transform(symbols_trans, start_set)
    np.linalg.norm((mts - reconst_same_shape).reshape(-1, np.prod(mts.shape[1:])), 'fro')


.. 

Note that ``jabba`` use init='agg' as default, one can set it to ``kmeans`` for improved performance while resulting in slower speed. If one switch to ``kmeans`` method, the hyperparameter of ``alpha`` and ``auto_digitize`` is disabled, instead of using them, one should tune the hyperparameter of ``k``, which refers to the number of clusters (distinct symbols) will be used. 

.. code:: python
 
    mts = np.random.randn(20, 20, 30) # new 6000 time series values
    
    # For aggregation, init='agg' is default
    jabba = JABBA(tol=0.01, alpha=0.01, verbose=1)
    symbols = jabba.fit_transform(mts)
    reconst = jabba.inverse_transform(symbols) # convert into array
    reconst_same_shape = jabba.recast_shape(reconst) # recast into original shape
    np.linalg.norm((mts - reconst_same_shape).reshape(-1, np.prod(mts.shape[1:])), 'fro')

    # For kmeans, init='k-means++'
    jabba = JABBA(tol=0.01, k=100, init='kmeans', verbose=1) # use 100 distinct symbols
    symbols = jabba.fit_transform(mts)
    reconst = jabba.inverse_transform(symbols) # convert into array
    reconst_same_shape = jabba.recast_shape(reconst) # recast into original shape
    np.linalg.norm((mts - reconst_same_shape).reshape(-1, np.prod(mts.shape[1:])), 'fro')


You can also load dataset via ``loadData``:

.. code:: python
    
    from fABBA import loadData
    train, test = loadData(name='Beef') 
    # Then perform JABBA
    jabba = JABBA(tol=0.0002, verbose=1)
    symbols_series = jabba.fit_transform(train[0])
    reconstruction = jabba.inverse_transform(symbols_series)

.. admonition:: Note
    
        function loadData() is a lightweight API for time series dataset loading, which only supports part of data in UEA or UCR Archive, please refer to the document for full use detail. JABBA is used to process multiple time series as well as multivariate time series, so the input should be ensured to be 2-dimensional, for example, when loading the UCI dataset, e.g., Beef, use symbols = jabba.fit_transform(train) , when loading UEA dataset, e.g., BasicMotions, use symbols = jabba.fit_transform(train[0]) . For details, we refer to `UCR/UEA time series dataset <https://www.timeseriesclassification.com/>`_.
        Functionality of ``loadData()`` currently supports datasets: (1) UEA Archive: 'AtrialFibrillation', 'BasicMotions', 'BasicMotions', 'CharacterTrajectories', 'LSST', 'Epilepsy', 'NATOPS', 'UWaveGestureLibrary', 'JapaneseVowels'; (2) UCR Archive: 'Beef'.
    
