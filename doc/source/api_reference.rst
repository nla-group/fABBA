API Reference
======================================

``fABBA`` is the API of the symbolic representation transformation for univariate time series.

fABBA
-------
.. automodule:: fABBA.symbolic_representation.fABBA
   :members:
 

``ABBA`` is the API of the symbolic representation transformation for univariate time series, which allows for ABBA method with various clustering techniques. 

ABBAbase
-------
.. automodule:: fABBA.symbolic_representation.ABBAbase
   :members:
   

``ABBA`` is the API of the symbolic representation ransformation for multivariate (rep., multiple univariate) time series.


JABBA
-------
.. automodule:: fABBA.jabba.jabba.JABBA
   :members:
   

We illustrate some main components of ``fABBA`` below.
 

compress
-------
.. autoclass:: fABBA.chainApproximation.compress
   :members:
   

inverse_compress
-------
.. autoclass:: fABBA.chainApproximation.inverse_compress
   :members:

   
digitize
-------
.. autoclass:: fABBA.digitization.digitize
   :members:
   

inverse_digitize
-------
.. autoclass:: fABBA.digitization.inverse_digitize
   :members:


We can employ image compressing with ``fABBA`` using the convenient API ``image_compress`` and ``image_decompress``.


image_compress
-------
.. autoclass:: fABBA.symbolic_representation.image_compress
   :members:



image_decompress
-------
.. autoclass:: fABBA.symbolic_representation.image_decompress
   :members:
