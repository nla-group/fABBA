.. CLASSIX documentation

Welcome to fABBA's documentation!
===================================
fABBA is a fast and accurate symbolic representation method for temporal data. It is based on a polygonal chain approximation of the time series followed by an aggregation of the polygonal pieces into groups. The aggregation process is sped up by sorting the polygonal pieces and exploiting early termination conditions. In contrast to the ABBA method [S. Elsworth and S. Güttel, Data Mining and Knowledge Discovery, 34:1175-1200, 2020], fABBA avoids repeated within-cluster-sum-of-squares computations which reduces its computational complexity significantly. Furthermore, fABBA is fully tolerance-driven and does not require the number of time series symbols to be specified by the user.

The `fABBA` softare package includes numerous ABBA methods by providing easy-to-use and flexible APIs. In practice, users can define their clustering method and apply it to ABBA method easily with `fABBA`.  The documentation is still under going, welcome to join our documentation building, please be free to pull request your contribution code!

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
