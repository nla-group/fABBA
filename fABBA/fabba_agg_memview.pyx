#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True

# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Cython implementation for aggregation for fABBA


cimport cython
import numpy as np
cimport numpy as np 
# from cython.parallel import prange
# from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
# from libc.string cimport strcmp
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
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    # cdef double[:] s1
    cdef double[:, :] cdata = np.empty((len_ind, fdim), dtype=np.float64)
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef Py_ssize_t sp # starting point index
    cdef unsigned int lab=0, num_group # , nr_dist=0
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef long[:] labels = np.full(len_ind, -1, dtype=int) 
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, ii, j, coord
    
    
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
    
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - data[j,coord])**2
            
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

