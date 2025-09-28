import numpy as np


def inv_transform(strings, centers, alphabets, start=0):
    """
    Convert ABBA symbolic representation back to numeric time series representation.

    Parameters
    ----------
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.

    centers - numpy array
        centers of clusters from clustering algorithm. Each centre corresponds
        to character in string.

    alphabets - list
        The alphabet set for symbols reversing.
        
    start - float
        First element of original time series. Applies vertical shift in
        reconstruction. If not specified, the default is 0.
        
    Returns
    -------
    times_series - list
        Reconstruction of the time series.
    """

    pieces = inv_digitize(strings, centers, alphabets)

    pieces = quantize(pieces)
    time_series = inv_compress(pieces, start)
    return time_series



def inv_transform_fp(strings, centers, alphabets, start=0):
    """
    Convert ABBA symbolic representation back to numeric time series representation.

    Parameters
    ----------
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.

    centers - numpy array
        centers of clusters from clustering algorithm. Each centre corresponds
        to character in string.

    alphabets - list
        The alphabet set for symbols reversing.
        
    start - float
        First element of original time series. Applies vertical shift in
        reconstruction. If not specified, the default is 0.
        
    Returns
    -------
    times_series - list
        Reconstruction of the time series.
    """

    pieces = inv_digitize(strings, centers, alphabets)

    pieces = quantize(pieces)
    time_series = inv_compress_fp(pieces, start)
    return time_series


def inv_digitize(strings, centers, alphabets):
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

    alphabets - list
        The alphabet set for symbols reversing.
        
    Returns
    -------
    pieces - np.array
        Time series in compressed format. See compression.
    """
    return np.vstack([centers[alphabets.index(p)][:2] for p in strings])



def quantize(pieces):
    """
    Realign window lengths with integer grid.

    Parameters
    ----------
    pieces: Time series in compressed representation.


    Returns
    -------
    pieces: Time series in compressed representation with window length adjusted to integer grid.
    """

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



def inv_compress(pieces, start):
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

    time_series = [start]
    # stitch linear piece onto last
    for j in range(0, len(pieces)):
        x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
        y = time_series[-1] + x
        time_series = time_series + y[1:].tolist()

    return time_series


def inv_compress_fp(pieces, start):
    """(fixed point)
    Reconstruct time series from its first value `ts0` and its `pieces`.
    `pieces` must have (at least) two columns, increment and window width, resp.
    A window width w means that the piece ranges from s to s+w.
    In particular, a window width of 1 is allowed.

    Parameters
    ----------
    pieces : numpy array
        Numpy array with three columns, each row contains increment, length,
        error for the segment. Only the first two columns are required.
    start : float
        First element of original time series. Applies vertical shift in
        reconstruction.

    Returns
    -------
    time_series : list
        Reconstructed time series
    """
    time_series = [start]
    
    # First piece
    x = np.arange(0, pieces[0, 0] + 1) / pieces[0, 0] * (pieces[0, 1] - start)
    y = start + x
    time_series.extend(y[1:].tolist())
    
    # Stitch linear pieces for remaining segments
    for j in range(1, len(pieces)):
        x = np.arange(0, pieces[j, 0] + 1) / pieces[j, 0] * (pieces[j, 1] - pieces[j - 1, 1])
        y = pieces[j - 1, 1] + x
        time_series.extend(y[1:].tolist())

    return time_series