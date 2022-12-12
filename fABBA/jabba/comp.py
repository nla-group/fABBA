import numpy as np

def compress(ts, tol=0.5, max_len=np.inf):
    """
    Approximate a time series using a continuous piecewise linear function.

    Parameters
    ----------
    ts - numpy ndarray
        Time series as input of numpy array

    Returns
    -------
    pieces - numpy array
        Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    """

    start = 0
    end = 1
    pieces = list() # np.empty([0, 3])
    x = np.arange(0, len(ts))
    epsilon =  np.finfo(float).eps
    
    temp = list()
    temp.append(ts[start])
    
    while end < len(ts):
        inc = ts[end] - ts[start]
        err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
        err = np.inner(err, err)
        
        if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < max_len):
            temp.append(ts[end])
            (lastinc, lasterr) = (inc, err) 
            end += 1
        else:
            pieces.append([end-start-1, lastinc, lasterr])
            start = end - 1
            temp = list()
            
    pieces.append([end-start-1, lastinc, lasterr])

    return pieces