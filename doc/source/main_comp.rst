
Main components
======================================


In this section, we mainly introduce the main components of transformation of ``fABBA`` for univariate time series. 

Adaptive polygonal chain approximation
------------------------------

Instead of using ``fit_transform`` which combines the polygonal chain approximation of the time series and the symbolic conversion into one, both steps of fABBA can be performed independently. Here’s how to obtain the compression pieces and reconstruct time series by inversely transforming the pieces:

.. code:: python

    import numpy as np
    from fABBA import compress
    from fABBA import inverse_compress
    ts = [np.sin(0.05*i) for i in range(1000)]
    pieces = compress(ts, tol=0.1)               # pieces is a list of the polygonal chain pieces
    inverse_ts = inverse_compress(pieces, ts[0]) # reconstruct polygonal chain from pieces



Symbolic digitization
------------------------------

Similarly, the fABBA digitization can be performed after compression step as belows:


.. code:: python

    from fABBA import digitize
    from fABBA import inverse_digitize
    string, parameters = digitize(pieces, alpha=0.1, sorting='2-norm', scl=1) # compression of the polygon
    print(''.join(string))                                 # prints BbAaAaAaAaAaAaAaC

    inverse_pieces = inverse_digitize(string, parameters)
    inverse_ts = inverse_compress(inverse_pieces, ts[0])   # numerical time series reconstruction



