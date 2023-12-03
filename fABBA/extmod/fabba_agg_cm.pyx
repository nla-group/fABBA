#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True

# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Cython implementation for aggregation for fABBA, limited to 2-dimensional data


cimport cython
import numpy as np
cimport numpy as np 
# from cython.parallel import prange
# from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
# from libc.string cimport strcmp
np.import_array()

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.binding(True)


cpdef aggregate(double[:,:] data, str sorting, double tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting way refered for aggregation, default='2-norm', alternative option: '1-norm' and 'lexi'.

    tol : float
        the tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  

    Returns
    -------
    labels (numpy.ndarray) : 
        the group category of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    nr_dist (int) :
        distance computation calculations
    """
    
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    # cdef double[:] s1
    cdef double[:, :] cdata = np.empty((len_ind, 2), dtype=np.float64)
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef Py_ssize_t sp # starting point index
    cdef unsigned int lab=0, num_group # , nr_dist=0
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef long[:] labels = np.full(len_ind, -1, dtype=int) 
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, ii, j
    
    
    if sorting == "2-norm":
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "1-norm":
        sort_vals = np.linalg.norm(data, ord=1, axis=1)
        ind = np.argsort(sort_vals)
    else:
        ind = np.lexsort((data[:,1], data[:,0]), axis=0) 

    # else: # no sorting
    #     ind = np.arange(len_ind)

    for i in range(len_ind): 
        sp = ind[i] # starting point
        
        if labels[sp] >= 0:
            continue
        
        clustc = data[sp,:] 
        labels[sp] = lab
        num_group = 1
            
        for ii in range(i, len_ind): 
            j = ind[ii]
                    
            if labels[j] != -1:
                continue
    

            dist = (clustc[0] - data[j, 0])**2
            dist += (clustc[1] - data[j, 1])**2
            
            # nr_dist += 1
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab
            else: # apply early stopping
                if sorting == "2-norm" or sorting == "1-norm":
                    if (sort_vals[j] - sort_vals[sp] > tol):
                        break       
                else:
                    if ((data[j,0] - data[sp,0] == tol) and (data[j,1] > data[sp,1])) or (data[j,0] - data[sp,0] > tol): 
                        break
                        
        splist.append( [sp, lab] + [num_group] + list(data[sp,:] ) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        
        lab += 1
        
    return np.asarray(labels), splist # , nr_dist

