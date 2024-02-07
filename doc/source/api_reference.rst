API Reference
======================================

``fABBA`` is the API of the symbolic representation transformation for univariate time series.

fABBA
-------
.. autoclass:: fABBA.symbolic_representation.fABBA
   :members:
 
.. autoclass:: fABBA.loadData
   :members:

.. autoclass:: fABBA.load_images
   :members:

``ABBA`` is the API of the symbolic representation transformation for univariate time series, which allows for ABBA method with various clustering techniques. 

ABBAbase
-------
.. autoclass:: fABBA.symbolic_representation.ABBAbase
   :members:
   

``ABBA`` is the API of the symbolic representation ransformation for multivariate (rep., multiple univariate) time series.


JABBA
-------
.. autoclass:: fABBA.jabba.jabba.JABBA
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
