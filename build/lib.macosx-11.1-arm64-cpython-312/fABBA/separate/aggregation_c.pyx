#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True


# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Cython implementation for aggregation  which is not limited to 2-dimensional data


cimport cython
import numpy as np
cimport numpy as np 
from scipy.sparse.linalg import svds
np.import_array()

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.binding(True)

cpdef aggregate(np.ndarray[np.float64_t, ndim=2] data, str sorting="norm", float tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting way refered for aggregation, default='norm', alternative option: 'pca'.

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
    
    cdef unsigned int num_group
    cdef unsigned int fdim = data.shape[1] # feature dimension
    cdef unsigned int len_ind = data.shape[0] # size of data
    cdef unsigned int sp # sp: starting point
    cdef unsigned int lab = 0 # lab: class
    cdef double dist # distance 
    cdef np.ndarray[np.int64_t, ndim=1] labels = np.zeros(len_ind, dtype=int) - 1
    cdef list splist = list() # store the starting points
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals = np.empty((len_ind, ), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] clustc = np.empty((fdim, ), dtype=float)
    cdef np.ndarray[np.int64_t, ndim=1] ind = np.empty((len_ind, ), dtype=int)
    cdef unsigned int i, j, coord, c
    
    if sorting == "norm": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    else: 
        data = data - np.mean(data, axis=0)
        if data.shape[1]>1:
            U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
        else:
            sort_vals = data[:,0]
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)


    for i in range(len_ind):
        sp = ind[i] 
        if labels[sp] >= 0:
            continue
        else:
            clustc = data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue
            
            if (sort_vals[j] - sort_vals[sp] > tol):
                break       
            
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - data[j,coord])**2
            
            # nr_dist += 1
            
            if dist <= tol**2:
                num_group = num_group + 1
                labels[j] = lab

        splist.append( [sp, lab] + [num_group] + list(data[sp,:]) ) # respectively store starting point
                                                                # index, label, number of neighbor objects, center (starting point).
        lab += 1

    return labels, splist #, nr_dist, agg_centers





# move to lite_func.py
# cpdef merge_pairs(list pairs):
#     """Transform connected pairs to connected groups (list)"""
# 
#     cdef list labels = list()
#     cdef list sub = list()
#     cdef Py_ssize_t i, j, maxid = 0
#     cdef Py_ssize_t len_p = len(pairs)
#     
#     cdef np.ndarray[np.int64_t, ndim=1] ulabels = np.full(len_p, -1, dtype=int) # np.zeros(len(pairs), dtype=int) - 1
#     cdef np.ndarray[np.int64_t, ndim=1] distinct_ulabels = np.unique(ulabels)
#     cdef np.ndarray[np.int64_t, ndim=1] select_arr
#     
#     for i in range(len_p):
#         if ulabels[i] == -1:
#             sub = pairs[i]
#             ulabels[i] = maxid
# 
#             for j in range(i+1, len_p):
#                 com = pairs[j]
#                 if check_if_intersect(sub, com):
#                     sub = sub + com
#                     if ulabels[j] == -1:
#                         ulabels[j] = maxid
#                     else:
#                         ulabels[ulabels == maxid] = ulabels[j]
# 
#             maxid = maxid + 1
# 
#     for i in distinct_ulabels:
#         sub = list()
#         select_arr = np.where(ulabels == i)[0]
#         for j in select_arr:
#             sub = sub + pairs[j]
#         labels.append(list(set(sub)))
#         
#     return labels



# cdef check_if_intersect(list g1, list g2):
#     return set(g1).intersection(g2) != set()
