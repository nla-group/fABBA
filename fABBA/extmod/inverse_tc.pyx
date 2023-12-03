cimport cython
import numpy as np
cimport numpy as np
np.import_array()

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.binding(True)

    
cpdef inv_transform(str strings, np.ndarray[np.float64_t, ndim=2] centers, list alphabets, double start=0):
    """
    Convert ABBA symbolic representation back to numeric time series representation.

    Parameters
    ----------
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.
    
    centers - numpy array
        Centers of clusters from clustering algorithm. Each center corresponds
        to character in string.

    alphabets - list
        Dictionary associated with labels and symbols.
        
    start - float
        First element of original time series. Applies vertical shift in
        reconstruction. If not specified, the default is 0.
        
    Returns
    -------
    times_series - list
        Reconstruction of the time series.
    """

    cdef np.ndarray[np.float64_t, ndim=2] pieces = inv_digitize(strings, centers, alphabets)

    pieces = quantize(pieces)
    cdef list time_series = inv_compress(pieces, start)
    return time_series



cpdef inv_digitize(str strings, np.ndarray[np.float64_t, ndim=2] centers, list alphabets):
    """
    Convert symbolic representation back to compressed representation for reconstruction.

    Parameters
    ----------
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.

    centers - numpy array
        centers of clusters from clustering algorithm. Each centre corresponds
        to character in string.

    alphabets - numpy.ndarray
        Dictionary associated with labels and symbols.

    Returns
    -------
    pieces - np.array
        Time series in compressed format. See compression.
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] pieces
    pieces = np.vstack([centers[alphabets.index(p)][:2] for p in strings])

    return pieces



cpdef quantize(np.ndarray[np.float64_t, ndim=2] pieces):
    """
    Realign window lengths with integer grid.

    Parameters
    ----------
    pieces: Time series in compressed representation.

    Returns
    -------
    pieces: Time series in compressed representation with window length adjusted to integer grid.
    """
    cdef double corr 
    cdef Py_ssize_t p
    
    if len(pieces) == 1:
        pieces[0,0] = round(pieces[0,0])

    else:
        for p in range(len(pieces)-1):
            corr = round(pieces[p,0]) - pieces[p,0]
            pieces[p,0] = round(pieces[p,0] + corr)
            pieces[p+1,0] = pieces[p+1,0] - corr
            if pieces[p,0] == 0:
                pieces[p,0] = 1
                pieces[p+1,0] -= 1
        pieces[-1,0] = round(pieces[-1,0],0)

    return pieces



cpdef inv_compress(np.ndarray[np.float64_t, ndim=2] pieces, double start):
    """
    Reconstruct time series from its first value `ts0` and its `pieces`.
    `pieces` must have (at least) two columns, incremenent and window width, resp.
    A window width w means that the piece ranges from s to s+w.
    In particular, a window width of 1 is allowed.

    Parameters
    ----------
    pieces - numpy array
        Numpy array with three columns, each row contains increment, length,
        error for the segment. Only the first two columns are required.

    start - float
        First element of original time series. Applies vertical shift in
        reconstruction.

    Returns
    -------
    time_series : Reconstructed time series
    """

    cdef list time_series = [start]
    cdef Py_ssize_t j
    cdef np.ndarray[np.float64_t, ndim=1] x, y
    
    # stitch linear piece onto last
    for j in range(0, len(pieces)):
        x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
        y = time_series[-1] + x
        time_series = time_series + y[1:].tolist()

    return time_series


