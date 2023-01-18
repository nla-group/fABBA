.. CLASSIX documentation

Welcome to fABBA's documentation!
===================================
ABBA is a fast and accurate symbolic aggregate approximation method for temporal data. It is based on a polygonal chain approximation of the time series followed by a mean clustering of the polygonal pieces into groups, which are associated with symbols assignment. 

With the symbolic aggregate approximation method, the time series is discretized into low-dimensional space after ABBA symbolization, and it naturally leads to noise smoothing, as well as improvement and acceleration for the deployment of most algorithms. moreover, there are numerous downstream time series applications as well as existing research work with symbolic representation, for example, time series compression, clustering, classification, and forecasting, most of which have shown superior performance with symbolic representation against the applications with raw time series. 

The fABBA method is a fast variant of ABBA. In contrast to the ABBA method [S. Elsworth and S. Güttel, Data Mining and Knowledge Discovery, 34:1175-1200, 2020], fABBA's digitization process is sped up by sorting the polygonal pieces and exploiting early termination conditions and avoids repeated within-cluster-sum-of-squares computations by using fast aggregation which reduces its computational complexity significantly. Furthermore, fABBA is fully tolerance-driven and does not require the number of time series symbols to be specified by the user. An illustration of the ABBA symbolization is given in the below picture from Stefan Güttel​'s talk  `in Turing-Manchester Project Presentations <https://www.youtube.com/watch?v=YEJLAYJ5SOA/>`_. 


.. image:: images/abba.png
    :width: 560
    
The `fABBA` software package includes numerous ABBA methods by providing easy-to-use and flexible APIs. In practice, users can define their clustering method and apply it to the ABBA method easily with `fABBA`.  `fABBA` not only allows for the symbolization of a single time series (resp., univariate time series) but also allows for the symbolization of multiple time series (resp., multivariate time series) with unified symbols information. 

The documentation provides a comprehensive review and guide for the usage of the package `fABBA`. Current documentation is still going, welcome to join our documentation building, please be free to pull request your contribution code!


* :ref:`search`

    
    
Guide
-------------

.. toctree::
   :maxdepth: 2
   
   quickstart
   main_comp
   multivariate
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
