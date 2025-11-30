#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True


# Cython implementation for aggregation


cimport cython
import numpy as np
cimport numpy as np 
from scipy.sparse.linalg import svds
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)


cpdef aggregate(double[:,:] data, str sorting, double tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting way refered for aggregation, default='pca', alternative option: 'pca'.

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
    
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    # cdef double[:] s1
    cdef double[:, :] cdata = np.empty((len_ind, fdim), dtype=np.float64)
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef Py_ssize_t sp # starting point index
    cdef unsigned int lab=0, num_group #, nr_dist=0
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef long[:] labels = np.full(len_ind, -1, dtype=int) 
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, ii, j, coord
    
    
    if sorting == "norm":
        cdata[...] = data
        sort_vals = np.linalg.norm(cdata, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    else: # sorting == "pca":
        # pca = PCA(n_components=1) 
        # sort_vals = pca.fit_transform(data_memview).reshape(-1)
        # ind = np.argsort(sort_vals)
        
        # change to svd 
        # cdata = data - data.mean(axis=0) -- already done in the clustering.fit_transform
        cdata = data - np.mean(data, axis=0)
        if data.shape[1]>1:
            U1, s1, _ = svds(cdata.base, k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
        else:
            sort_vals = cdata[:,0]
            
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    # else: # no sorting
    #     ind = np.arange(len_ind)


    for i in range(len_ind): 
        sp = ind[i] # starting point
        
        if labels[sp] >= 0:
            continue
        
        clustc = cdata[sp,:] 
        labels[sp] = lab
        num_group = 1
            
        for ii in range(i, len_ind): 
            j = ind[ii]
                    
            if labels[j] != -1:
                continue
                
            if (sort_vals[j] - sort_vals[sp] > tol):
                break       
            
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - cdata[j,coord])**2
            # nr_dist += 1

            if dist <= tol**2:
                num_group += 1
                labels[j] = lab
                
        splist.append( [sp, lab] + [num_group] + list(data[sp,:] ) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        
        lab += 1
        
    return np.asarray(labels), splist #, nr_dist



cpdef aggregate_1d(double[:] data, double tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting method for aggregation, default='norm', alternative option: 'pca'.

    tol : float
        the tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    Returns
    -------
    labels (numpy.ndarray) : 
        the group categories of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    *nr_dist (int) :
    *    number of pairwise distance calculations
    """

    cdef list splist = list() # store the starting points
    cdef double[:] sort_vals = np.squeeze(data)
    cdef double dat, clustc
    cdef int len_ind = len(sort_vals)
    cdef long[:] ind = np.argsort(sort_vals) # order by increasing size
    
    cdef Py_ssize_t lab = 0
    cdef list labels = [-1]*len_ind
    cdef Py_ssize_t nr_dist = 0 
    cdef Py_ssize_t sp, num_group, i, j
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            # clustc = data[sp]
            if data[sp] < sort_vals[-1] - tol:
                clustc = data[sp] + tol
            else:
                clustc = data[sp]
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue
            
            if (np.abs(sort_vals[j] - clustc) > tol):
                break       
                
            dat = clustc - data[j]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, lab] + [num_group] + [clustc]) 
        # respectively store starting point
        # index, label, number of neighbor objects, center (starting point).
        lab += 1
    return np.array(labels), splist

