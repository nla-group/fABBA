# Copyright (c) 2021, 
# Authors: Stefan Güttel, Xinye Chen

# All rights reserved.


import copy
import pickle
import warnings
import logging
import collections
import numpy as np
import pandas as pd
from functools import wraps
from dataclasses import dataclass
from sklearn.cluster import KMeans
from inspect import signature, Parameter

try:
    try:# cython with memory view
        from .separate.aggregation_cm import aggregate as aggregate_fc 
        from .extmod.chainApproximation_cm import compress
        import platform
        
        if platform.system() != 'Windows':
            from .extmod.fabba_agg_cm import aggregate as aggregate_fabba 
        else:
            from .extmod.fabba_agg_cm_win import aggregate as aggregate_fabba 
        
    except ModuleNotFoundError:
        from .extmod.chainApproximation_c import compress
        from .separate.aggregation_c import aggregate as aggregate_fc 
        from .extmod.fabba_agg_c import aggregate as aggregate_fabba 
        warnings.warn("Installation is not using Cython typed memoryviews.")
    
    from .extmod.inverse_tc import *
    
    
except (ModuleNotFoundError):
    from .chainApproximation import compress
    from .separate.aggregation import aggregate as aggregate_fc 
    from .fabba_agg import aggregate as aggregate_fabba
    from .inverse_t import *
    warnings.warn("This installation is not using Cython.")



class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    """



@dataclass
class Model:
    """
    save ABBA model - parameters
    """
    centers: np.ndarray # store aggregation centers
    splist: np.ndarray # store start point data
    
    """ dictionary """
    alphabets: np.ndarray # labels -> symbols, symbols -> labels





class Aggregation2D:
    """ A separatate aggregation for data with 2-dimensional (2D) features. 
        Independent applicable to 2D data aggregation
        
    Parameters
    ----------
    alpha - float, default=0.5
        Control tolerence for digitization        
    
    sorting - str, default='2-norm', {'lexi', '1-norm', '2-norm'}
        by which the sorting pieces prior to aggregation
    
    """
    
    def __init__(self, alpha=0.5, sorting='2-norm'):
        self.alpha = alpha
        self.sorting = sorting
        
        
        
    def aggregate(self, data):
                
        if self.sorting == 'lexi':
            ind = np.lexsort((data[:,1], data[:,0]), axis=0) 
        
        elif self.sorting == '2-norm':
            ind = np.argsort(np.linalg.norm(data, ord=2, axis=1))
        
        elif self.sorting == '1-norm':
            ind = np.argsort(np.linalg.norm(data, ord=1, axis=1))
            
        lab = 0
        splist = list() 
        labels = 0*ind - 1
        
        for i in range(len(ind)):
            sp = ind[i]
            
            if labels[sp] >= 0:
                continue
            
            else:
                clustc = data[sp,:] 
                labels[sp] = lab
                splist.append([sp, lab] + list(clustc))
                
                if self.sorting == '2-norm':
                    center_norm = np.linalg.norm(clustc, ord=2)
                
                elif self.sorting == '1-norm':
                    center_norm = np.linalg.norm(clustc, ord=1)

            for j in ind[i:]:
                if labels[j] >= 0:
                    continue

                if self.sorting == 'lexi':
                    if ((data[j,0] - data[sp,0] == self.alpha)\
                        and (data[j,1] > data[sp,1])) or (data[j,0] - data[sp,0] > self.alpha): 
                        break
                        
                elif self.sorting == '2-norm':
                    if np.linalg.norm(data[j,:], ord=2, axis=0) - center_norm > self.alpha: 
                        break
                        
                elif self.sorting == '1-norm':
                    if 0.707101 * (np.linalg.norm(data[j,:], ord=1, axis=0) - center_norm) > self.alpha: 
                        break

                dist = np.sum((clustc - data[j,:])**2) 
                
                if dist <= self.alpha**2:
                    labels[j] = lab
            
            lab += 1

        return labels, np.array(splist)

    
    
    
def _deprecate_positional_args(func=None, *, version=None):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    
    Paste from: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/utils/validation.py#L1034
    
    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
        
    version : callable, default="1.0 (renaming of 0.25)"
        The version when positional arguments will result in error.
    """
    
    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = ['{}={}'.format(name, arg)
                        for name, arg in zip(kwonly_args[:extra_args],
                                             args[-extra_args:])]
            args_msg = ", ".join(args_msg)
            warnings.warn(f"Pass {args_msg} as keyword args. From next version "
                          f"{version} passing these as positional arguments "
                          "will result in an error", FutureWarning)
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)
        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args



    
def image_compress(fabba, data, adjust=True):
    """ image compression. """
    ts = data.reshape(-1)
    if adjust:
        _mean = ts.mean(axis=0)
        _std = ts.std(axis=0)
        if _std == 0:
            _std = 1
        ts = (ts - _mean) / _std
        string = fabba.fit_transform(ts)
        fabba.img_norm = (_mean, _std)
    else:
        fabba.img_norm = None
        string = fabba.fit_transform(ts)
    fabba.img_start = ts[0]
    fabba.img_shape = data.shape
    return string



def image_decompress(fabba, string):
    """ image decompression. """
    reconstruction = np.array(fabba.inverse_transform(string, start=fabba.img_start))
    if fabba.img_norm != None:
        reconstruction = reconstruction*fabba.img_norm[1] + fabba.img_norm[0]
    reconstruction = reconstruction.round().reshape(fabba.img_shape).astype(np.uint8)
    return  reconstruction


   
def _compress(series, tol=0.5, max_len=-1, fillm='bfill'):
    """
    Compress time series.

    Parameters
    ----------
    series - numpy.ndarray or list
        Time series of the shape (1, n_samples).
    
    tol - float
        The tolerance that controls the accuracy.
    
    max_len - int
        The maximum length that compression restriction.
        
    fillm - str, default = 'zero'
        Fill NA/NaN values using the specified method.
        'Zero': Fill the holes of series with value of 0.
        'Mean': Fill the holes of series with mean value.
        'Median': Fill the holes of series with mean value.
        'ffill': Forward last valid observation to fill gap.
            If the first element is nan, then will set it to zero.
        'bfill': Use next valid observation to fill gap. 
            If the last element is nan, then will set it to zero.   

    """
    
    series = np.array(series).astype(np.float64)
    if len(series.shape) > 1:
        series = series.reshape(-1)
        
    if np.sum(np.isnan(series)) > 0:
        series = fillna(series, fillm)
    
    return compress(ts=series, tol=tol, max_len=max_len)




def _inverse_compress(pieces, start):
    pieces = np.array(pieces)[:, :2]
    return inv_compress(pieces, start)




def symbolsAssign(clusters, alphabet_set=0):
    """
    Automatically assign symbols to different groups, start with '!'
    
    Parameters
    ----------
    clusters - list or pd.Series or array
        The list of labels.
            
    alphabet_set - int or list
        The list of alphabet letter.
        
    ----------
    Return:
    
    string (list of string), alphabets(numpy.ndarray): for the
    corresponding symbolic sequence and for mapping from symbols to labels or 
    labels to symbols, repectively.

    """
    clusters = pd.Series(clusters)
    N = len(clusters.unique())
    
    if alphabet_set == 0:
        alphabets = ['A','a','B','b','C','c','D','d','E','e',
                    'F','f','G','g','H','h','I','i','J','j',
                    'K','k','L','l','M','m','N','n','O','o',
                    'P','p','Q','q','R','r','S','s','T','t',
                    'U','u','V','v','W','w','X','x','Y','y','Z','z']
    
    elif alphabet_set == 1:
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                    'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
                    'w', 'x', 'y', 'z']
    
    elif isinstance(alphabet_set, list):
        if N <= len(alphabet_set):
            alphabets = alphabet_set
        else:
            raise ValueError("Please ensure the length of ``alphabet_set`` is greatere than ``clusters``.")
       
    else:
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                    'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z']
        
    cluster_sort = [0] * N 
    counter = collections.Counter(clusters)
    for ind, el in enumerate(counter.most_common()):
        cluster_sort[ind] = el[0]

    if N >= len(alphabets):
        alphabets = [chr(i+33) for i in range(0, N)]
    else:
        alphabets = alphabets[:N]

    alphabets = np.asarray(alphabets)
    string = alphabets[clusters]
    return string, alphabets


    
    
    

class ABBAbase:
    def __init__ (self, clustering, tol=0.1, scl=1, verbose=1, max_len=-1):
        """
        This class is designed for other clustering based ABBA
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        scl - int
            Scale for length, default as 1, means 2d-digitization, otherwise implement 1d-digitization.
        verbose - int
            Control logs print, default as 1, print logs.
        max_len - int
            The max length for each segment, default as -1. 
        
        """
        
        self.tol = tol
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        # self.compress = compress
        self.compression_rate = None
        self.digitization_rate = None
        self.clustering = clustering
        
        
    def fit(self, series, fillm='bfill', alphabet_set=0):
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
            
        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero.   
        """

        if np.sum(np.isnan(series)) > 0:
            series = fillna(series, method=fillm)
        series = np.array(series).astype(np.float64)
        pieces = np.array(self.compress(series))
        self.string_, self.parameters = self.digitize(pieces[:,0:2], alphabet_set)
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.parameters.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(self.string_)), "to", self.parameters.centers.shape[0], "symbols.")  
        self.string_ = ''.join(self.string_)
        return self
    

    def fit_transform(self, series, fillm='bfill', alphabet_set=0):
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
            
        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero.   
        """
        
        return self.fit(series, fillm, alphabet_set).string_
    
    
    
    def inverse_transform(self, string, start=0, parameters=None):
        """
        Convert ABBA symbolic representation back to numeric time series representation.
        
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction. If not specified, the default is 0.
        
        parameters - Model
            The parameters of model.
            
            
        Returns
        -------
        series - list
            Reconstruction of the time series.
        """
        if type(string) != str:
            string = "".join(string)
            
        if parameters is None:
            try:
                series = inv_transform(string, self.parameters.centers, self.parameters.alphabets.tolist(), start) 
            except:
                raise NotFittedError("Please train the model using ``fit_transform`` first.")
        else:
            series = inv_transform(string, parameters.centers, parameters.alphabets.tolist(), start) 
        return series

    
    
    
    def compress(self, series, fillm='bfill'):
        """
        Compress time series.
        
        Parameters
        ----------
        series - numpy.ndarray or list
            Time series of the shape (1, n_samples).

        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero.   
        
        """
        
        return _compress(series=np.array(series).astype(np.float64), tol=self.tol, max_len=self.max_len, fillm=fillm)
    
    
    
    def digitize(self, pieces, alphabet_set=0):
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
        pieces = np.array(pieces)[:,:2]
        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[1] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces * np.array([self.scl, 1]) / _std
        
        # replace aggregation with other clustering
        labels = self.reassign_labels(self.clustering.fit_predict(npieces)) # some labels might be negative
        centers = np.zeros((0,2))
        for c in range(len(np.unique(labels))):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
            
        # self.centers = centers
        string, alphabets = symbolsAssign(labels, alphabet_set)
        parameters = Model(centers, centers, alphabets)
        return string, parameters


    
    def reassign_labels(self, labels):
        old_labels_count = collections.Counter(labels)
        sorted_dict = sorted(old_labels_count.items(), key=lambda x: x[1], reverse=True)

        clabels = copy.deepcopy(labels)
        for i in range(len(sorted_dict)):
            clabels[labels == sorted_dict[i][0]]  = i
        return clabels
    
    
    
    # def inverse_transform(self, string, start=0):
    #     pieces = self.inverse_digitize(string, self.parameters.centers, self.parameters.alphabets)
    #     pieces = self.quantize(pieces)
    #     series = self.inverse_compress(pieces, start)
    #     return series
    # 
    # 
    # def inverse_digitize(self, string, centers, alphabetsap):
    #     pieces = np.empty([0,2])
    #     for p in string:
    #         pc = centers[int(alphabetsap[p])]
    #         pieces = np.vstack([pieces, pc])
    #     return pieces[:,0:2]
    # 
    # 
    # def quantize(self, pieces):
    #     if len(pieces) == 1:
    #         pieces[0,0] = round(pieces[0,0])
    #     else:
    #         for p in range(len(pieces)-1):
    #             corr = round(pieces[p,0]) - pieces[p,0]
    #             pieces[p,0] = round(pieces[p,0] + corr)
    #             pieces[p+1,0] = pieces[p+1,0] - corr
    #             if pieces[p,0] == 0:
    #                 pieces[p,0] = 1
    #                 pieces[p+1,0] -= 1
    #         pieces[-1,0] = round(pieces[-1,0],0)
    #     return pieces

    
    
    
class ABBA(ABBAbase):
    def __init__ (self, tol=0.1, k=2, scl=1, verbose=1, max_len=-1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0, verbose=0)    
        super().__init__(clustering=kmeans, tol=tol, scl=scl, verbose=verbose, max_len=max_len)
        
    def digitize(self, pieces, alphabet_set=0):
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
        pieces = np.array(pieces)[:,:2]
        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[1] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces * np.array([self.scl, 1]) / _std
        
        # replace aggregation with other clustering
        self.clustering.fit(np.unique(npieces, axis=0))
        
        labels = self.reassign_labels(self.clustering.fit_predict(npieces)) # some labels might be negative
        centers = np.zeros((0,2))
        for c in range(len(np.unique(labels))):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
            
        # self.centers = centers
        string, alphbets = symbolsAssign(labels, alphabet_set)
        parameters = Model(centers, centers, alphbets)
        return string, parameters

    
    
    
def get_patches(ts, pieces, string, centers, dictionary):
    """
    Follow original ABBA smooth reconstruction, 
    creates a dictionary of patches from time series data using the clustering result.
    
    Parameters
    ----------
    ts - numpy array
        Original time series.
        
    pieces - numpy array
        Time series in compressed format.
        
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.
        
    centers - numpy array
        Centers of clusters from clustering algorithm. Each centre corresponds
        to a character in string.
        
    ditionary - dict
         For mapping from symbols to labels or labels to symbols.
        
    
    Returns
    -------
    patches - dict
        A dictionary of time series patches.
    """
    
    pieces = np.array(pieces)
    patches = dict()
    inds = 0
    for j in range(len(pieces)):
        symbol = string[j]                        # letter
        lab = dictionary[symbol]                  # label (integer)
        lgt = round(centers[lab,0])               # patch length
        inc = centers[lab,1]                      # patch increment
        inde = inds + int(pieces[j,0]);
        tsp = ts[inds:inde+1]                      # time series patch

        tsp = tsp - (tsp[-1]-tsp[0]-inc)/2-tsp[0]  # shift patch so that it is vertically centered with patch increment

        tspi = np.interp(np.linspace(0,1,lgt+1), np.linspace(0,1,len(tsp)), tsp)
        if symbol in patches:
            patches[symbol] = np.append(patches[symbol], np.array([tspi]), axis = 0)
        else:
            patches[symbol] = np.array([ tspi ])
        inds = inde
    return patches



def patched_reconstruction(series, pieces, string, centers, dictionary):
    """
    An alternative reconstruction procedure which builds patches for each
    cluster by extrapolating/intepolating the segments and taking the mean.
    The reconstructed time series is no longer guaranteed to be of the same
    length as the original.
    
    Parameters
    ----------
    series - numpy array
        Normalised time series as numpy array.
        
    pieces - numpy array
        One or both columns from compression. See compression.
        
    string - string
        Time series in symbolic representation using unicode characters starting
        with character 'a'.
        
    centers - numpy array
        centers of clusters from clustering algorithm. Each center corresponds
        to character in string.

    ditionary - dict
         For mapping from symbols to labels or labels to symbols.
    """
    if type(string) is list:
        string = "".join(string)
         
    patches = get_patches(series, pieces, string, centers, dictionary)
    # Construct mean of each patch
    d = {}
    for key in patches:
        d[key] = list(np.mean(patches[key], axis=0))

    reconstructed_series = [series[0]]
    for letter in string:
        patch = d[letter]
        patch -= patch[0] - reconstructed_series[-1] # shift vertically
        reconstructed_series = reconstructed_series + patch[1:].tolist()
    return reconstructed_series



class fABBA(Aggregation2D, ABBAbase):
    """
    fABBA: A fast sorting-based aggregation method for symbolic time series representation
    
    Parameters
    ----------
    tol - float, default=0.1
        Control tolerence for compression.
    
    alpha - float, default=0.5
        Control tolerence for digitization.        
    
    sorting - str, default='2-norm', {'lexi', '1-norm', '2-norm'}
        by which the sorting pieces prior to aggregation.
        
    scl - int, default=1
        Scale for length, default as 1, refers to 2d-digitization, otherwise implement 1d-digitization.
    
    verbose - int, default=1
        Verbosity mode, control logs print, default as 1; print logs.
    
    max_len - int, default=-1
        The max length for each segment, optional choice for compression.
    
    return_list - boolean, default=True
        Whether to return with list or not, "False" means return string.
    
    n_jobs - int, default=-1 
        The number of threads to use for the computation.
        -1 means no parallel computing.
        
    
    Attributes
    ----------
    parameters - Model
        Contains the learnable parameters from the in-sample data. 
        
        Attributes:
        * centers - numpy.ndarray
            the centers calculated for each group formed by aggregation
        * splist - numpy.ndarray
            the starting point for each group formed by aggregation
        * alphabetsap - dict
            store the oen to one key-value pair for labels earmarked for the groups
            and the corresponding character
    
    string_ - str or list
        Contains the ABBA representation.

    
    * In addition to fit_transform, the compression and digitization functions are independent applicable to data. 
    """    
    
    def __init__ (self, tol=0.1, alpha=0.5, 
                  sorting='2-norm', scl=1, verbose=1,
                  max_len=-1, return_list=False, n_jobs=1):
        
        super().__init__()
        self.tol = tol
        self.alpha = alpha
        self.sorting = sorting
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        self.return_list = return_list
        self.n_jobs = n_jobs # For the moment, we don't use this parameter.
        # self.compress = compress

        
        
    def __repr__(self):
        parameters_dict = self.__dict__.copy()
        parameters_dict.pop('_std', None)
        parameters_dict.pop('logger', None)
        parameters_dict.pop('parameters', None)
        parameters_dict.pop('compress', None)
        parameters_dict.pop('n_jobs', None) # For the moment, we don't use this parameter.
        return "%s(%r)" % ("fABBA", parameters_dict)

    
    
    def __str__(self):
        parameters_dict = self.__dict__.copy()
        parameters_dict.pop('_std', None)
        parameters_dict.pop('logger', None)
        parameters_dict.pop('parameters', None)
        parameters_dict.pop('compress', None)
        parameters_dict.pop('n_jobs', None) # For the moment, we don't use this parameter.
        return "%s(%r)" % ("fABBA", parameters_dict)
    
    
    def fit(self, series, fillm='bfill', alphabet_set=0):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - numpy.ndarray or list
            Time series of the shape (1, n_samples).
            
        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero. 
                
        Returns
        ----------
        string (str): The string transformed by fABBA.
        """
        
        if np.sum(np.isnan(series)) > 0:
            series = fillna(series, fillm)

        # if self.n_jobs > 1 and self.max_len == 1:
        #     pieces = self.parallel_compress(ts=series, n_jobs=self.n_jobs)
        # else:
        #     # pieces = self.compress(ts=series)
        pieces = self.compress(series)
            
        self.string_, self.parameters = self.digitize(
            pieces=np.array(pieces)[:,0:2], alphabet_set=alphabet_set
        )
        
        if self.verbose:
            _info = "Digitization: Reduced pieces of length {}".format(
                len(self.string_)) + " to {} ".format(len(self.parameters.centers)) + " symbols"
            self.logger.info(_info)

        if not self.return_list:
            self.string_ = "".join(self.string_)
            
        return self
    

    def fit_transform(self, series, fillm='bfill', alphabet_set=0):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - numpy.ndarray or list
            Time series of the shape (1, n_samples).
            
        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero. 
                
        Returns
        ----------
        string (str): The string transformed by fABBA.
        """
        return self.fit(series, fillm, alphabet_set).string_



    def inverse_transform(self, string, start=0, parameters=None):
        """
        Convert ABBA symbolic representation back to numeric time series representation.
        
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction. If not specified, the default is 0.
        
        parameters - Model
            The parameters of model.
            
        Returns
        -------
        series - list
            Reconstruction of the time series.
        """
        
        if type(string) != str:
            string = "".join(string)
            
        if parameters is None:
            try:
                series = inv_transform(string, self.parameters.centers, self.parameters.alphabets.tolist(), start) 
            except:
                raise NotFittedError("Please train the model using ``fit_transform`` first.") 
        else:
            series = inv_transform(string, parameters.centers, parameters.alphabets.tolist(), start) 
    
        return series
    
    
    
    # deprecated
    # def compress(self, ts):
    #     """
    #     Approximate a time series using a continuous piecewise linear function.
    #     
    #     Parameters
    #     ----------
    #     ts - numpy ndarray
    #         Time series as input of numpy array
    # 
    #     Returns
    #     -------
    #     pieces - numpy array
    #         Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    #     """
    #     
    #     start = 0
    #     end = 1
    #     pieces = np.empty([0, 3])
    #     x = np.arange(0, len(ts))
    #     epsilon =  np.finfo(float).eps
    # 
    #     while end < len(ts):
    #         inc = ts[end] - ts[start]
    #         err = np.linalg.norm((ts[start] + (inc/(end-start))*x[0:end-start+1]) - ts[start:end+1])**2
    #         
    #         if (err <= self.tol*(end-start-1) + epsilon) and (end-start-1 < self.max_len):
    #             (lastinc, lasterr) = (inc, err) 
    #             end += 1
    #         else:
    #             pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    #             start = end - 1
    # 
    #     pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    #     
    #     if self.verbose:
    #         self.logger = logging.getLogger("fABBA")
    #         self.logger.info(
    #             "Compression: Reduced time series of length "  
    #             + str(len(ts)) + " to " + str(len(pieces)) + " segments")
    #         
    #     return pieces

    

    # def parallel_compress(self, series, n_jobs=-1):
    #     """
    #     Approximate a time series using a continuous piecewise linear function in a parallel way.
    #     Each piece is of length 1. 
    #     
    #     Parameters
    #     ----------
    #     series - numpy ndarray
    #         Time series as input of numpy array
    #     
    #         
    #     Returns
    #     -------
    #     pieces - numpy array
    #         Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    #     """
    #     from joblib import Parallel, delayed
    #     x = np.arange(0, len(series))
    # 
    #     def construct_piece(i):
    #         inc = series[i+1] - series[i]
    #         err = np.linalg.norm((series[i] + (inc)*x[0:2]) - series[i:i+2])**2
    #         return [1, inc, err]
    # 
    #     pieces = Parallel(n_jobs=n_jobs)(
    #         delayed(construct_piece)(i) for i in range(len(series) - 1))
    # 
    #     if self.verbose:
    #         self.logger = logging.getLogger("fABBA")
    #         self.logger.info(
    #             "Compression: Reduced time series of length "  
    #             + str(len(series)) + " to " + str(len(pieces)) + " segments")
    # 
    #     return np.array(pieces)


    def compress(self, series, fillm='bfill'):
        """
        Compress time series.
        
        Parameters
        ----------
        series - numpy.ndarray or list
            Time series of the shape (1, n_samples).

        fillm - str, default = 'zero'
            Fill NA/NaN values using the specified method.
            'Zero': Fill the holes of series with value of 0.
            'Mean': Fill the holes of series with mean value.
            'Median': Fill the holes of series with mean value.
            'ffill': Forward last valid observation to fill gap.
                If the first element is nan, then will set it to zero.
            'bfill': Use next valid observation to fill gap. 
                If the last element is nan, then will set it to zero.   
        
        """
        
        return _compress(series=np.array(series).astype(np.float64), tol=self.tol, max_len=self.max_len, fillm=fillm)
    
    
    
    @_deprecate_positional_args
    def digitize(self, pieces, alphabet_set=0):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce alpha and len/inc scaling parameter scl.
        A 'temporary' group center, which we call it starting point,
        is used  when assigning pieces to clusters. This temporary
        cluster is the first piece available after appropriate scaling 
        and sorting of all pieces. After finishing the grouping procedure,
        the centers are calculated the mean value of the objects within 
        the clusters.
        
        Parameters
        ----------
        pieces - numpy.ndarray
            The compressed pieces of numpy.ndarray with shape (n_samples, n_features) after compression.
            
        alphabet_set - int or list
            The list of alphabet letter.
        
        Returns
        ----------
        string - str or list)
            String sequence.
            
        parameters - Model
            The parameters of model.
        """

        if self.sorting not in ["lexi", "2-norm", "1-norm", "norm", "pca"]:
            raise ValueError("Please refer to a specific and correct sorting way, namely 'lexi', '2-norm' and '1-norm'")
        
        pieces = np.array(pieces)[:,:2].astype(np.float64)
        self._std = np.std(pieces, axis=0) 
        
        if self._std[0] != 0: # to prevent 0 std when assign max_len as 1 to compression, which make aggregation go wrong.
            npieces = pieces * np.array([self.scl, 1]) / self._std
        else:
            npieces = pieces * np.array([self.scl, 1])
            npieces[:,1] = npieces[:,1] / self._std[1]
        
        if self.sorting in ["lexi", "2-norm", "1-norm"]:
            # warnings.warn(f"Pass {self.sorting} as keyword args. From the next version ", FutureWarning)
            labels, splist = aggregate_fabba(npieces, self.sorting, self.alpha)
        else:
            labels, splist = aggregate_fc(npieces, self.sorting, self.alpha)

        centers = np.zeros((0,2))
        
        for c in range(len(splist)):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
        
        string, alphabets = symbolsAssign(labels, alphabet_set)
        
        parameters = Model(centers, np.array(splist), alphabets)
        return string, parameters

    
    
    # [DEPRECATED]
    # def inverse_transform(self, string, parameters=None, start=0):
    #     """
    #     Convert ABBA symbolic representation back to numeric time series representation.
    #     
    #     Parameters
    #     ----------
    #     string - string
    #         Time series in symbolic representation using unicode characters starting
    #         with character 'a'.
    #     
    #     start - float
    #         First element of original time series. Applies vertical shift in
    #         reconstruction. If not specified, the default is 0.
    #     
    #     Returns
    #     -------
    #     times_series - list
    #         Reconstruction of the time series.
    #     """
    #
    #     if parameters == None:
    #         pieces = self.inverse_digitize(string, self.parameters)
    #     else:
    #         pieces = self.inverse_digitize(string, parameters)
    #         
    #     pieces = self.quantize(pieces)
    #     series = self.inverse_compress(pieces, start)
    #     return series
    # 
    # 
    # 
    # def inverse_digitize(self, string, parameters):
    #     """
    #     Convert symbolic representation back to compressed representation for reconstruction.
    #     
    #     Parameters
    #     ----------
    #     string - string
    #         Time series in symbolic representation using unicode characters starting
    #         with character 'a'.
    #         
    #     centers - numpy array
    #         centers of clusters from clustering algorithm. Each centre corresponds
    #         to character in string.
    #         
    #     Returns
    #     -------
    #     pieces - np.array
    #         Time series in compressed format. See compression.
    #     """
    #     
    #     pieces = np.empty([0,2])
    #     for p in string:
    #         pc = parameters.centers[int(parameters.inverse_alphabets[p])]
    #         pieces = np.vstack([pieces, pc])
    #     return pieces[:,0:2]
    # 
    # 
    # 
    # def quantize(self, pieces):
    #     """
    #     Realign window lengths with integer grid.
    #     
    #     Parameters
    #     ----------
    #     pieces: Time series in compressed representation.
    #     
    #     
    #     Returns
    #     -------
    #     pieces: Time series in compressed representation with window length adjusted to integer grid.
    #     """
    #         
    #     if len(pieces) == 1:
    #         pieces[0,0] = round(pieces[0,0])
    #     
    #     else:
    #         for p in range(len(pieces)-1):
    #             corr = round(pieces[p,0]) - pieces[p,0]
    #             pieces[p,0] = round(pieces[p,0] + corr)
    #             pieces[p+1,0] = pieces[p+1,0] - corr
    #             if pieces[p,0] == 0:
    #                 pieces[p,0] = 1
    #                 pieces[p+1,0] -= 1
    #         pieces[-1,0] = round(pieces[-1,0],0)
    #     
    #     return pieces
    # 
    # 
    # 
    # def inverse_compress(self, pieces, start):
    #     """
    #     Reconstruct time series from its first value `ts0` and its `pieces`.
    #     `pieces` must have (at least) two columns, incremenent and window width, resp.
    #     A window width w means that the piece ranges from s to s+w.
    #     In particular, a window width of 1 is allowed.
    #     
    #     Parameters
    #     ----------
    #     start - float
    #         First element of original time series. Applies vertical shift in
    #         reconstruction.
    #     
    #     pieces - numpy array
    #         Numpy array with three columns, each row contains increment, length,
    #         error for the segment. Only the first two columns are required.
    #     
    #     Returns
    #     -------
    #     series : Reconstructed time series
    #     """
    #     
    #     series = [start]
    #     # stitch linear piece onto last
    #     for j in range(0, len(pieces)):
    #         x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
    #         y = series[-1] + x
    #         series = series + y[1:].tolist()
    # 
    #     return series
    
                
    # save model
    def dump(self, file=None):
        if file == None:
            pickle.dump(self.parameters, open("parameters", "wb"))
        else:
            pickle.dump(self.parameters, open(file, "wb"))
        
        
    # load model
    def load(self, file=None, replace=False):
        if file == None:
            parameters = pickle.load(open("parameters", "rb"))
        else:
            parameters = pickle.load(open(file, "rb"))
            
        if replace:
            self.parameters = parameters
            print("load completed.")
        else:
            return parameters
        
        
        
    @staticmethod
    def print_parameters(cls):
        print("Centers:")
        print(cls.parameters.centers)
        print("\nalphabetsap:")
        for i, item in enumerate(cls.parameters.alphabets.items()):
            print(item)

            
    
    @property
    def tol(self):
        return self._tol
    
    
    
    @tol.setter
    def tol(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected a float or int type.")
        if value <= 0:
            raise ValueError(
                "Please feed an correct value for tolerance.")
        if value > 1:
            warnings.warn("Might lead to bad aggregation.", DeprecationWarning)
        self._tol = value
    
    
        
    @property
    def sorting(self):
        return self._sorting
    
    
    
    @sorting.setter
    def sorting(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected a string type")
        if value not in ["lexi", "2-norm", "1-norm", "norm", "pca"]:
            raise ValueError(
                "Please refer to an correct sorting way, namely 'lexi', '2-norm' and '1-norm'.")
        self._sorting = value

    

    @property
    def scl(self):
        return self._scl



    @scl.setter
    def scl(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for scl.")
        
        if value > 1:
            warnings.warn("Might lead to bad aggregation.", DeprecationWarning)
        
        self._scl = value

 

    @property
    def verbose(self):
        return self._verbose



    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected a float or int type.")
        
        self._verbose  = value
        if self.verbose == 1:
            self.logger = logging.getLogger("fABBA")
            logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
        


    @property
    def alpha(self):
        return self._alpha
    
    
    
    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected a float or int type.")
        
        if value <= 0:
            raise ValueError(
                "Please feed an correct value for alpha.")
        
        self._alpha = value



    @property
    def max_len(self):
        return self._max_len



    @max_len.setter
    def max_len(self, value):
        # if value == np.inf:
        #     if not isinstance(value, float) and not isinstance(value,int):
        #         raise TypeError("Expected a float or int type.")
        # 
        # if value <= 0:
        #     raise ValueError(
        #         "Please feed an correct value for max_len.")
        if value == np.inf:
            raise ValueError("Please feed an correct value for max_len.")
        self._max_len = value



    @property
    def return_list(self):
        return self._return_list



    @return_list.setter
    def return_list(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean type.")
        self._return_list = value


    @property
    def n_jobs(self):
        return self._n_jobs
    
    
    
    @n_jobs.setter
    def n_jobs(self, value):
        if not isinstance(value, int):
            raise TypeError("Expected a int type.")
        
        self._n_jobs = value

        

def fillna(series, method='zero'):
    """Fill the NA values
    
    Parameters
    ----------   
    series - numpy.ndarray or list
        Time series of the shape (1, n_samples).

    fillna - str, default = 'zero'
        Fill NA/NaN values using the specified method.
        'Zero': Fill the holes of series with value of 0.
        'Mean': Fill the holes of series with mean value.
        'Median': Fill the holes of series with mean value.
        'ffill': Forward last valid observation to fill gap.
            If the first element is nan, then will set it to zero.
        'bfill': Use next valid observation to fill gap. 
            If the last element is nan, then will set it to zero.        
    """

    if method == 'Mean':
        series[np.isnan(series)] = np.mean(series[~np.isnan(series)])

    elif method == 'Median':
        series[np.isnan(series)] = np.median(series[~np.isnan(series)])

    elif method == 'ffill':
        for i in np.where(np.isnan(series))[0]:
            if i > 0:
                series[i] = series[i-1]
            else:
                series[i] = 0

    elif method == 'bfill':
        for i in sorted(np.where(np.isnan(series))[0], reverse=True):
            if i < len(series):
                series[i] = series[i+1]
            else:
                series[i] = 0
    else:
        series[np.isnan(series)] = 0

    return series




