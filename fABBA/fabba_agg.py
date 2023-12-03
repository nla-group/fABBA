# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

#Python implementation for aggregation


import numpy as np

# python implementation for aggregation
def aggregate(data, sorting="2-norm", tol=0.5): # , verbose=1
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting method for aggregation, default='2-norm', alternative option: '1-norm' and 'lexi'.

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
    
    nr_dist (int) :
        number of pairwise distance calculations
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    if sorting == "2-norm": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "1-norm": 
        sort_vals = np.linalg.norm(data, ord=1, axis=1)
        ind = np.argsort(sort_vals)
    else:
        ind = np.lexsort((data[:,1], data[:,0]), axis=0) 
        
    lab = 0
    labels = [-1]*len_ind
    # nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            clustc = data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            dat = clustc - data[j,:]
            dist = np.inner(dat, dat)
            # nr_dist += 1
                
            if dist <= tol**2:
                # num_group += 1
                labels[j] = lab
            else: # apply early stopping
                if sorting == "2-norm" or sorting == "1-norm":
                    if (sort_vals[j] - sort_vals[sp] > tol):
                        break       
                else:
                    if ((data[j,0] - data[sp,0] == tol) and (data[j,1] > data[sp,1])) or (data[j,0] - data[sp,0] > tol): 
                        break
        splist.append([sp, lab] + [num_group] + list(data[sp,:]) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        lab += 1

    return np.array(labels), splist #, nr_dist 



# An independent aggregation code, deprecated in the experiment
# import logging
# import warnings
# import numpy as np
# from tqdm import tqdm 

# logger = logging.getLogger("aggregation")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")


# class Aggregation:
#     """
#     A fast aggregation based on greedily selecting cluster core.
    
#     Parameters
#     ----------
#     tol - float, default=0.5
#         Control tolerence for digitization, is equivalent to alpha in the paper.    
    
#     sorting - str, default='2-norm', {'lexi', '1-norm', '2-norm'}
#         by which the sorting pieces prior to aggregation.
    
#     verbose - int, default=1
#         Verbosity mode.
        
#     Attributes
#     ----------
#     agg_centers - numpy.ndarray
        
#     agg_sps - numpy.ndarray
#         storing the 
    
    
#     Examples
#     --------
#     >>> from aggregation import Aggregation
#     >>> import numpy as np
#     >>> np.random.seed(1)
#     >>> N = 100
#     >>> x = np.random.rand(N)
#     >>> y = np.random.rand(N)
#     >>> data = np.vstack((x, y)).T
#     >>> agg = Aggregation(sorting='1-norm', tol=0.5, verbose=0)
#     >>> agg.aggregate(data)
#     >>> print(agg.labels)
#     [0. 3. 2. 0. 2. 2. 0. 2. 2. 3. 0. 1. 2. 3. 0. 3. 2. 3. 2. 0. 1. 1. 0. 1.
#      3. 3. 0. 2. 0. 1. 2. 3. 3. 1. 3. 0. 3. 1. 2. 3. 3. 1. 0. 1. 0. 0. 1. 2.
#      2. 0. 0. 3. 2. 0. 1. 2. 1. 2. 3. 3. 0. 1. 3. 2. 0. 1. 1. 1. 3. 3. 1. 2.
#      2. 1. 2. 2. 3. 0. 1. 1. 3. 1. 3. 0. 0. 3. 0. 3. 3. 3. 0. 1. 2. 3. 2. 2.
#      1. 1. 2. 3.]

#     """
    
    
#     __slots__ = ['sorting', 'tol', 'agg_centers', 'agg_sps', 'verbose', 'labels']
    
#     def __init__(self, sorting, tol, verbose=1):
#         self.sorting = sorting
#         self.tol = tol
#         self.verbose = verbose
        
        
        
#     def __repr__(self):
#         return 'Aggregation(tol={0.tol!r},sorting={0.sorting!r},verbose={0.verbose!r})'.format(self)

    
    
#     def __str__(self):
#         return 'Aggregation(tol={0.tol!r},sorting={0.sorting!r},verbose={0.verbose!r})'.format(self)
    
    
    
#     def _check_params(self):
#         if not isinstance(self.tol, float) and not isinstance(self.tol,int):
#             raise TypeError('Expected a float or int type')
#         if self.tol <= 0:
#             raise ValueError(
#                 "Please feed an correct value (>0) for tolerance")
#         if self.tol > 1:
#             warnings.warn("Might lead to bad aggregation", DeprecationWarning)
        
#         if not isinstance(self.sorting, str):
#             raise TypeError('Expected a string type')
#         if self.sorting not in ['lexi', '2-norm', '1-norm']:
#             raise ValueError(
#                 "Please refer to an correct sorting way, namely 'lexi', '2-norm' and '1-norm'")
        
#         if not isinstance(self.verbose, float) and not isinstance(self.verbose,int):
#             raise TypeError('Expected a float or int type')

#         return 
        
        
        
#     def aggregate(self, data):
#         """aggregate the data
        
#         Parameters
#         ----------
#         data : numpy.ndarray
#             the input that is array-like of shape (n_samples,)
#         sorting : str
#             the sorting way refered for aggregation, default='2-norm', other options: '1-norm', 'lexi'.
#         tol : float
#             the tolerance to control the aggregation, if the distance between the starting point 
#             and the object is less than or equal than the tolerance,
#             the object should allocated to the group which starting point belongs to.
        
#         Returns
#         -------
#         self.labels (numpy.ndarray) : the group category of the data after aggregation.
        
#         self.agg_sps (list) : the list of the starting points.
#         """
        
#         self._check_params()
#         # data = (data - data.mean()) / np.std(data) 
#         ell = data.shape[1] # feasture dimensions
#         splist = list() # store the starting points

#         if self.sorting == 'lexi':
#             com = tuple()
#             for i in range(ell):
#                 com = (data[:,i],) + com
#             ind = list(np.lexsort(com , axis=0)); del com

#         elif self.sorting == '2-norm':
#             ind = np.argsort(np.linalg.norm(data, ord=2, axis=1))
#         else:
#             ind = np.argsort(np.linalg.norm(data, ord=1, axis=1))

#         lab = 0
#         len_ind = len(ind)
#         labels = np.zeros(len_ind) - 1
        
#         for i in tqdm(range(len_ind), disable=False):
#             sp = ind[i] # starting point
#             if labels[sp] >= 0:
#                 continue
#             else:
#                 clustc = data[sp,:] 
#                 labels[sp] = lab
#                 splist.append([sp, lab] + list(clustc))

#             if self.sorting == '2-norm':
#                 center_norm = np.linalg.norm(clustc, ord=2)
#             elif self.sorting == '1-norm':
#                 center_norm = np.linalg.norm(clustc, ord=1)

#             for j in ind[i:]:
#                 if labels[j] >= 0:
#                     continue
                
#                 if self.sorting == 'lexi':
#                     if ((data[j,0] - data[sp,0] == self.tol) and (data[j,1] > data[sp,1])) \
#                         or (data[j,0] - data[sp,0] > self.tol): 
#                         logger.debug("Early stopping applies: {}".format((sp, j)))
#                         break
#                 elif self.sorting == '2-norm':
#                     if np.linalg.norm(data[j,:], ord=2) - center_norm > self.tol: 
#                         logger.debug("Early stopping applies: {}".format((sp, j)))
#                         break
#                 else:
#                     if 1/ell * (np.linalg.norm(data[j,:], ord=1) - center_norm) > self.tol: 
#                         logger.debug("Early stopping applies: {}".format((sp, j)))
#                         break
                        
#                 dist = np.sum((clustc - data[j,:])**2) 
#                 if dist <= self.tol**2:
#                     labels[j] = lab

#             lab = lab + 1
               
#         self.agg_centers = np.zeros((0,data.shape[1]))
#         for c in range(lab):
#             indc = np.argwhere(labels==c)
#             center = np.mean(data[indc,:], axis=0)
#             self.agg_centers = np.r_[ self.agg_centers, center ]
        
#         if self.verbose:
#             logger.info("Generate {} clusters".format(lab))
            
#         self.agg_sps = np.array(splist)
#         self.labels = labels
        
#         return
