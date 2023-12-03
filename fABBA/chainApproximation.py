# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Adaptive polygonal chain approximation

import numpy as np

def compress(ts, tol=0.5, max_len=-1):
    """
    Approximate a time series using a continuous piecewise linear function.

    Parameters
    ----------
    ts - numpy ndarray
        Time series as input of numpy array.
    
    tol - float
        The tolerance that controls the accuracy.
    
    max_len - int
        The maximum length that compression restriction.
        
    Returns
    -------
    pieces - numpy array
        Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    """
    if max_len < 0:
        max_len = len(ts)
        
    start = 0
    end = 1
    pieces = list() # np.empty([0, 3])
    x = np.arange(0, len(ts))
    epsilon =  np.finfo(float).eps

    while end < len(ts):
        inc = ts[end] - ts[start]
        err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
        err = np.inner(err, err)

        if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < max_len):
            (lastinc, lasterr) = (inc, err) 
            end += 1
        else:
            # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
            pieces.append([end-start-1, lastinc, lasterr])
            start = end - 1

    # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    pieces.append([end-start-1, lastinc, lasterr])
    return pieces


def inverse_compress(pieces, start):
    """
    Reconstruct time series from its first value `ts0` and its `pieces`.
    `pieces` must have (at least) two columns, incremenent and window width, resp.
    A window width w means that the piece ranges from s to s+w.
    In particular, a window width of 1 is allowed.

    Parameters
    ----------
    start - float
        First element of original time series. Applies vertical shift in
        reconstruction.

    pieces - numpy array
        Numpy array with three columns, each row contains increment, length,
        error for the segment. Only the first two columns are required.

    Returns
    -------
    time_series : Reconstructed time series
    """
    
    pieces = np.array(pieces)
    time_series = [start]
    # stitch linear piece onto last
    for j in range(0, len(pieces)):
        x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
        y = time_series[-1] + x
        time_series = time_series + y[1:].tolist()

    return time_series