#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# We use this scipt to compare other clustering methods.
# A simple framework implementation for experiment use.


import copy
import warnings
import collections
import numpy as np
import pandas as pd
import collections
# warnings.filterwarnings('ignore')

try:
    # %load_ext Cython
    # !python3 setup.py build_ext --inplace
    from .compress_c import compress
    from .cagg_memview import aggregate as aggregate_fc # cython with memory view
except ModuleNotFoundError:
    warnings.warn("cython fail.")
    from .compress import compress
    from .agg import aggregate as aggregate_fc

class fABBA:
    def __init__ (self, clustering, tol=0.1, scl=1, verbose=1, max_len=np.inf):
        """
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        scl - int
            Scale for length, default as 1, means 2d-digitization, otherwise implement 1d-digitization.
        verbose - int
            Control logs print, default as 1, print logs.
        max_len - int
            The max length for each segment, default as np.inf. 
        
        """
        
        self.tol = tol
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        self.compress = compress
        self.compression_rate = None
        self.digitization_rate = None
        self.clustering = clustering
        
# (deprecated)
#     def compress(self, ts, tol=0.5, verbose=1, max_len=np.inf):
#         """
#         Approximate a time series using a continuous piecewise linear function.
#         
#         Parameters
#         ----------
#         ts - numpy array
#             Time series as numpy array.
#             
#         Returns
#         -------
#         pieces - numpy array
#             Numpy array with three columns, each row contains length, increment, error for the segment.
#         """
#         
#         start = 0
#         end = 1
#         pieces = np.empty([0, 3])
#         x = np.arange(0, len(ts))
#         epsilon =  np.finfo(float).eps
# 
#         while end < len(ts):
#             inc = ts[end] - ts[start]
#             err = np.linalg.norm((ts[start] + (inc/(end-start))*x[0:end-start+1]) - ts[start:end+1])**2
#             
#             if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < max_len):
#                 (lastinc, lasterr) = (inc, err) 
#                 end += 1
#                 continue
#             else:
#                 pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
#                 start = end - 1
# 
#         pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
#         if self.verbose in [1, 2]:
#             print('Compression: Reduced time series of length', len(ts), 'to', len(pieces), 'segments')
#         return pieces


    def image_compress(self, data, adjust=False):
        ts = data.reshape(-1)
        if adjust:
            _mean = ts.mean(axis=0)
            _std = ts.std(axis=0)
            if _std == 0:
                _std = 1
            ts = (ts - _mean) / _std
            strings = self.fit_transform(ts)
            self.img_norm = (_mean, _std)
        else:
            self.img_norm = None
            strings = self.fit_transform(ts)
        return strings, ts[0], self

    
    def image_decompress(self, strings, start, shape):
        reconstruction = np.array(self.inverse_transform(strings, start))
        if self.img_norm != None:
            reconstruction = reconstruction*self.img_norm[1] + self.img_norm[0]
        reconstruction = reconstruction.round().reshape(shape).astype(np.uint8)
        return  reconstruction
    
    
    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - array or list
            Time series.
        alpha - float
            Control tolerence for digitization, default as 0.5.
        string_form - boolean
            Whether to return with string form, default as True.
        """
        series = np.array(series).astype(np.float64)
        pieces = np.array(self.compress(ts=series, tol=self.tol, max_len=self.max_len))
        strings = self.digitize(pieces[:,0:2])
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(strings)), "to", self.centers.shape[0], "symbols.")  
        strings = ''.join(strings)
        return strings
    
    
    def digitize(self, pieces, early_stopping=True):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce tol and len/inc scaling parameter scl.

        In this variant, a 'temporary' cluster center is used 
        when assigning pieces to clusters. This temporary cluster
        is the first piece available after appropriate scaling 
        and sorting of all pieces. It is *not* necessarily the 
        mean of all pieces in that cluster and hence the final
        cluster centers, which are just the means, might achieve 
        a smaller within-cluster tol.
        """
        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[1] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces * np.array([self.scl, 1]) / _std
        # labels, self.splist, self.nr_dist = aggregate_fc(npieces, self.sorting, self.alpha)
        # replace aggregation with HDBSCAN
        labels = self.reassign_labels(self.clustering(npieces)) # some labels might be negative
        centers = np.zeros((0,2))
        for c in range(len(np.unique(labels))):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
        self.centers = centers
        strings, self.hashmap = self.symbolsAssign(labels)
        return strings
            
            
# (deprecated)
#    def digitize(self, pieces, sorting='2-norm', alpha=0.5, scl=1):
#         """
#         Greedy 2D clustering of pieces (a Nx2 numpy array),
#         using tolernce tol and len/inc scaling parameter scl.
# 
#         In this variant, a 'temporary' cluster center is used 
#         when assigning pieces to clusters. This temporary cluster
#         is the first piece available after appropriate scaling 
#         and sorting of all pieces. It is *not* necessarily the 
#         mean of all pieces in that cluster and hence the final
#         cluster centers, which are just the means, might achieve 
#         a smaller within-cluster tol.
#         """
#
#         if sorting not in ['lexicographic', '2-norm', '1-norm']:
#             raise ValueError("Please refer to a specific and correct sorting way, namely 'lexicographic', '2-norm' and '1-norm'")
#         npieces = pieces * np.array([scl, 1]) / np.std(pieces, axis=0) 
#         
#         if sorting == 'lexicographic':
#             ind = np.lexsort((npieces[:,1], npieces[:,0]), axis=0) 
#         elif sorting == '2-norm':
#             ind = np.argsort(np.linalg.norm(npieces, ord=2, axis=1))
#         else:
#             ind = np.argsort(np.linalg.norm(npieces, ord=1, axis=1))
#             
#         lab = 0; cnt = 0 # cnt counts the distance computations.
#         labels = 0*ind - 1
#         for i in range(len(ind)):
#             sp = ind[i]
#             if labels[sp] >= 0:
#                 continue
#             else:
#                 clustc = npieces[sp,:] 
#                 labels[sp] = lab     
#                 if sorting == '2-norm':
#                     center_norm = np.linalg.norm(clustc, ord=2)
#                 else: # for the cases of 1-norm
#                     center_norm = np.linalg.norm(clustc, ord=1)
# 
#             for j in ind[i:]:
#                 if labels[j] >= 0:
#                     continue
#                     
#                 if sorting == 'lexicographic':
#                     d = np.sum((clustc - npieces[j,:])**2)
#                     if ((npieces[j,0] - npieces[sp,0] == alpha)\
#                         and (npieces[j,1] > npieces[sp,1])) or (npieces[j,0] - npieces[sp,0] > alpha): 
#                         break
#                     
#                 elif sorting == '2-norm':
#                     d = np.sum((clustc - npieces[j,:])**2)
#                     if np.linalg.norm(npieces[j,:], ord=2, axis=0) - center_norm > alpha: 
#                         break
#                         
#                 elif sorting == '1-norm':
#                     d = np.linalg.norm(clustc - npieces[j,:], ord=1)**2
#                     if np.linalg.norm(npieces[j,:], ord=1, axis=0) - center_norm > alpha: 
#                         break
#                         
#                 else:
#                     d = np.sum((clustc - npieces[j,:])**2)
#                     if 0.707101 * (np.linalg.norm(npieces[j,:], ord=1, axis=0) - center_norm) > alpha: # when specify distance computed as 2 norm 
#                         break                                                                          # the early stopping refinement will degenerate
#                                                             # to this case
#                 
#                 cnt = cnt + 1
#                 
#                 if d <= alpha**2:
#                     labels[j] = lab
#             lab += 1
# 
#         centers = np.zeros((0,2))
#         for c in range(lab):
#             indc = np.argwhere(labels==c)
#             center = np.mean(pieces[indc,:], axis=0)
#             centers = np.r_[ centers, center ]
#         self.centers = centers
#         string, self.hashmap = self.symbolsAssign(labels)
#         if self.verbose in [1, 2]:
#             print('Digitization: Reduced pieces of length', len(string), 'to', len(self.centers), 'symbols')
#         return string, cnt

    
    def symbolsAssign(self, clusters):
        """ automatically assign symbols to different clusters, start with '!'

        Parameters
        ----------
        clusters(list or pd.Series or array): the list of clusters.

        -------------------------------------------------------------
        Return:
        symbols(list of string), inverse_hash(dict): repectively for corresponding symbols and hashmap for inverse transform.
        """
        
        clusters = pd.Series(clusters)
        N = len(clusters.unique())

        cluster_sort = [0] * N 
        counter = collections.Counter(clusters)
        for ind, el in enumerate(counter.most_common()):
            cluster_sort[ind] = el[0]

        alphabet= [chr(i) for i in range(33,33 + N)]
        hashmap = dict(zip(cluster_sort + alphabet, alphabet + cluster_sort))
        strings = [hashmap[i] for i in clusters]
        return strings, hashmap

    
    def reassign_labels(self, labels):
        old_labels_count = collections.Counter(labels)
        sorted_dict = sorted(old_labels_count.items(), key=lambda x: x[1], reverse=True)

        clabels = copy.deepcopy(labels)
        for i in range(len(sorted_dict)):
            clabels[labels == sorted_dict[i][0]]  = i
        return clabels
    
    
    def inverse_transform(self, strings, start=0):
        pieces = self.inverse_digitize(strings, self.centers, self.hashmap)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(pieces, start)
        return time_series

    
    def inverse_digitize(self, strings, centers, hashmap):
        pieces = np.empty([0,2])
        for p in strings:
            pc = centers[int(hashmap[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]

    
    def quantize(self, pieces):
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

    
    def inverse_compress(self, pieces, start):
        """Modified from ABBA package, please see ABBA package to see guidance."""
        
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            #print(x)
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()

        return time_series
    
    
    
    
# (deprecated, distorted figure)
# from tqdm import tqdm

# def image_compress(data):
#     start_points= list()
#     fabba_mapping = list()
#     strings = list()
#     shape = data[0].shape[:2]
#     squeeze = data.reshape(data.shape[0]*data.shape[1],-1).T
#     
#     for i in tqdm(range(squeeze.shape[0])):
#         ts = squeeze[i]
#         fabba = fABBA(tol=0.1, scl=1, sorting="2-norm", alpha=0.1, verbose=1, max_len=np.inf) 
#         strings.append(fabba.fit_transform(ts))
#         start_points.append(ts[0])
#         fabba_mapping.append(fabba)
#     return strings, start_points, fabba_mapping
# 
# def image_decompress(strings, start_points, fabba_mapping, shape=None):
#     reconstruction = list()
#     for i in range(len(strings)):
#         string = strings[i]
#         inverse_ts = fabba_mapping[i].inverse_transform(string, start_points[i])
#         reconstruction.append(inverse_ts)
#     if shape == None:
#         warnings.warn("Shape is none, which may lead to failure.")
#         shape = (int(np.sqrt(len(inverse_ts))), int(np.sqrt(len(inverse_ts))))
#     reconstruction = np.array(reconstruction).reshape(shape[0], shape[1], -1).astype(np.uint8)
#     return  reconstruction