
Get Started with fABBA
======================================


Installation guide
------------------------------
fABBA has the following essential dependencies for its functionality:

    * cython
    * numpy
    * scipy>=1.2.1
    * requests
    * scikit-learn
    * matplotlib


I. **pip**

To install the current release via PIP use:

.. parsed-literal::
    
    pip install fabba

To check the installation, simply run:

.. parsed-literal::
    
    python -m pip show fabba
    
If you want to uninstall it, you can use:

.. parsed-literal::

    pip uninstall fabba
    
II. **conda**

For conda users, to install this package with conda run:

.. parsed-literal::

    conda install -c conda-forge fabba
    
To check the installation, run:

.. parsed-literal::
    
    conda list fabba

and uninstall it with 

.. parsed-literal::

    conda uninstall fabba
   
   

Installing `fABBA` from the `conda-forge` channel can also be achieved by adding `conda-forge` to your channels with:

.. parsed-literal::

   conda config --add channels conda-forge
   conda config --set channel_priority strict

Once the `conda-forge` channel has been enabled, `fABBA` can be installed with `conda`:

.. parsed-literal::

   conda install fabba


or with `mamba`:

.. parsed-literal::

   mamba install fabba


It is possible to list all of the versions of `fABBA` available on your platform with `conda`:

.. parsed-literal::

   conda search fabba --channel conda-forge


or with `mamba`:

.. parsed-literal::

   mamba search fabba --channel conda-forge


Alternatively, `mamba repoquery` may provide more information:

.. parsed-literal::

   # Search all versions available on your platform:
   mamba repoquery search fabba --channel conda-forge

   # List packages depending on `fABBA`:
   mamba repoquery whoneeds fabba --channel conda-forge

   # List dependencies of `fABBA`:
   mamba repoquery depends fabba --channel conda-forge



III. **download**

Download this repository via:

.. parsed-literal::
    
    git clone https://github.com/nla-group/fABBA.git

If you have any instaling issues, please be free to submit your questions in the `issues <https://github.com/nla-group/fABBA/issues>`_.


Quick start
------------------------------



The following example approximately transforms a time series into a symbolic string representation (`fit_transform`) and then converts the string back into a numerical format (`inverse_transform`). fABBA essentially requires two parameters `tol` and `alpha`. The tolerance `tol` determines how closely the polygonal chain approximation follows the original time series. The parameter `alpha` controls how similar time series pieces need to be in order to be represented by the same symbol. A smaller `tol` means that more polygonal pieces are used and the polygonal chain approximation is more accurate; but on the other hand, it will increase the length of the string representation. A smaller `alpha` typically results in a larger number of symbols. 

The choice of parameters depends on the application, but in practice, one often just wants the polygonal chain to mimic the key features in time series and not to approximate any noise. In this example the time series is a sine wave and the chosen parameters result in the symbolic representation `BbAaAaAaAaAaAaAaC`. Note how the periodicity in the time series is nicely reflected in repetitions in its string representation.


.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from fABBA import fABBA

    ts = [np.sin(0.05*i) for i in range(1000)]  # original time series
    fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

    string = fabba.fit_transform(ts)            # string representation of the time series
    print(string)                               # prints BbAaAaAaAaAaAaAaC

    inverse_ts = fabba.inverse_transform(string, ts[0]) # numerical time series reconstruction

.. admonition:: Remember
    

Now you can plot your reconstruction to see how close it is to the raw data:

.. code:: python

    plt.plot(ts, label='time series', c='olive')
    plt.plot(inverse_ts, label='reconstruction', c='darkblue')
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()



.. image:: images/demo.png

