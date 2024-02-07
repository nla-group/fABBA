API Reference
======================================

``fABBA`` is the API of the symbolic representation transformation for univariate time series.

fABBA
-------
.. autoclass:: fABBA.fABBA
   :members:
 
.. autoclass:: fABBA.loadData
   :members:

.. autoclass:: fABBA.load_images
   :members:



ABBAbase
-------
.. autoclass:: fABBA.ABBAbase
   :members:
   

JABBA
-------

``JABBA`` is the API of the symbolic representation transformation for univariate time series, multivariate (rep., multiple univariate) time series, which allows for the combination of ABBA method with various clustering techniques. 


.. autoclass:: fABBA.JABBA
   :members:
   

We illustrate some main components of ``fABBA`` below.
 

compress
-------
.. autoclass:: fABBA.chainApproximation.compress
   :members:
   

inverse_compress
-------
.. autoclass:: fABBA.compress
   :members:

   
digitize
-------
.. autoclass:: fABBA.digitize
   :members:
   

inverse_digitize
-------
.. autoclass:: fABBA.inverse_digitize
   :members:


We can employ image compressing with ``fABBA`` using the convenient API ``image_compress`` and ``image_decompress``.


image_compress
-------
.. autoclass:: fABBA.image_compress
   :members:



image_decompress
-------
.. autoclass:: fABBA.image_decompress
   :members:
