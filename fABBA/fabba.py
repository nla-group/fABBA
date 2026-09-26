# Copyright (c) 2021, 
# Authors: Stefan Güttel, Xinye Chen

# All rights reserved.


import copy
import pickle
import warnings
import logging
import collections
import os
from ._validation import nonnegative, max_length, series_array, piece_array
import numpy as np
import pandas as pd
from functools import wraps
from dataclasses import dataclass
from sklearn.cluster import KMeans
from inspect import signature, Parameter
from multiprocessing.pool import ThreadPool as Pool
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
    """Learned codebook: centers are unscaled ``[length, increment]`` rows.

    ``alphabets[i]`` names ``centers[i]``. ``splist`` contains backend-specific
    aggregation diagnostics; it is not required for decoding.
    """
    centers: np.ndarray
    splist: np.ndarray
    alphabets: np.ndarray

    def to_dict(self):
        """Return an independent JSON-compatible, versioned codebook."""
        return {"schema_version": 1, "centers": self.centers.tolist(),
                "splist": self.splist.tolist(), "alphabets": self.alphabets.tolist()}

    @classmethod
    def from_dict(cls, data):
        """Validate and restore a codebook exported by :meth:`to_dict`."""
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported codebook schema_version (expected 1)")
        try:
            centers = piece_array(data["centers"])
            alphabets = np.asarray(data["alphabets"])
            splist = np.asarray(data.get("splist", []), dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid codebook arrays") from exc
        if (alphabets.ndim != 1 or len(alphabets) != len(centers)
                or any(not isinstance(x, str) or len(x) != 1 for x in alphabets.tolist())
                or len(set(alphabets.tolist())) != len(alphabets)):
            raise ValueError("alphabets must contain one unique character per center")
        if not np.isfinite(splist).all():
            raise ValueError("splist must be finite")
        return cls(centers, splist.copy(), alphabets.copy())





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
    
    series = series_array(series)
    tol = nonnegative(tol, "tol")
    max_len = max_length(max_len)
    series = fillna(series, fillm)
    return compress(ts=series, tol=tol, max_len=max_len)





def _inverse_compress(pieces, start):
    from .inverse_t import inv_compress as reconstruct
    return reconstruct(piece_array(pieces), start)




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
            if (any(not isinstance(x, str) or len(x) != 1 for x in alphabet_set)
                    or len(set(alphabet_set)) != len(alphabet_set)):
                raise ValueError("alphabet_set must contain unique single characters")
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

    if N > len(alphabets):
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

        series = fillna(series_array(series), fillm)
        pieces = np.array(self.compress(series, fillm=fillm))
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
        if parameters is None:
            if not hasattr(self, "parameters"):
                raise NotFittedError("Call fit or fit_transform before decoding.")
            parameters = self.parameters
        from .inverse_t import inv_transform as reconstruct
        return reconstruct(string, parameters.centers, parameters.alphabets.tolist(), start)

    
    
    
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
        
        return _compress(series=series, tol=self.tol, max_len=self.max_len, fillm=fillm)
    
    
    
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
             _std[0] = 1
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
             _std[0] = 1
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
    """Tolerance-driven symbolic approximation of a univariate time series.

    Parameters
    ----------
    tol : float, default=0.1
        Nonnegative polygonal squared-error tolerance per interior sample.
    alpha : float, default=0.5
        Nonnegative grouping radius in scaled length/increment coordinates.
    sorting : str, default='2-norm'
        One of 'lexi', '1-norm', '2-norm', 'norm' or 'pca'.
    scl : float, default=1
        Nonnegative relative weight of the segment-length feature.
    verbose : int, default=1
        Enable logging. Configure logging handlers in the calling application.
    partition_rate : float or None, default=None
        Optional positive rate used to derive a partition count.
    partition : int or None, default=None
        Optional explicit positive partition count; overrides partition_rate.
    fillna : str, default='ffill'
        Missing-value policy for the partition helper. Public fit/compress
        methods accept their own explicit fillm argument.
    max_len : int, default=-1
        Maximum segment length in intervals; -1 means unlimited.
    return_list : bool, default=False
        Return a symbol array instead of a joined string.
    n_jobs : int, default=1
        Worker count for partitioned compression; -1 selects available CPUs.

    Attributes
    ----------
    parameters : Model
        Learned centers, alphabet and aggregation diagnostics.
    string_ : str or numpy.ndarray
        Symbol representation of the fitted signal.
    pieces_ : numpy.ndarray
        Polygonal rows [length, increment, squared_error].
    start_ : float
        First sample after missing-value filling.
    n_samples_ : int
        Number of fitted samples.

    Notes
    -----
    Reconstruction is lossy. The compression tolerance is not a bound on
    full-pipeline reconstruction error. Use JABBA for held-out encoding with
    a shared codebook; calling fit again learns a new codebook.
    """
    
    def __init__ (self, tol=0.1, alpha=0.5, 
                  sorting='2-norm', scl=1, verbose=1, 
                  partition_rate=None, partition=None, fillna='ffill', 
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
        self.fillna = fillna

        self.partition = partition
        self.partition_rate = partition_rate
        self.return_series_univariate = None
        
        
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
        """Learn a codebook and retain the training representation; return self.

        Parameters
        ----------
        series : array-like
            Real univariate signal with at least two samples.
        fillm : str, default='bfill'
            NaN policy: zero, mean, median, ffill or bfill. Input is copied.
        alphabet_set : int or list of str, default=0
            Built-in ordering (0 or 1), or unique single-character symbols.
        """
        
        series = fillna(series_array(series), fillm)
        pieces = self.compress(series, fillm=fillm)
        self.start_ = float(series[0])
        self.n_samples_ = len(series)
        self.pieces_ = np.asarray(pieces, dtype=float)

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
        """Fit a new codebook and return a string (or symbol array).

        Arguments match fit. The learned codebook is available in parameters;
        this method does not return a (symbols, centers) tuple.
        """
        return self.fit(series, fillm, alphabet_set).string_



    def inverse_transform(self, string, start=0, parameters=None):
        """Decode symbols using the fitted or explicitly supplied codebook.

        Parameters
        ----------
        string : str or sequence of str
            Symbols in the associated alphabet. Empty input returns [start].
        start : float, default=0
            First value in the reconstructed signal.
        parameters : Model or None, default=None
            Explicit codebook; if omitted, the estimator must be fitted.

        Returns
        -------
        list of float
            Lossy reconstructed signal. Edited symbol sequences can have a
            different duration from the original fitted signal.
        """
        
        if parameters is None:
            if not hasattr(self, "parameters"):
                raise NotFittedError("Call fit or fit_transform before decoding.")
            parameters = self.parameters
        from .inverse_t import inv_transform as reconstruct
        return reconstruct(string, parameters.centers, parameters.alphabets.tolist(), start)
    
    
    
    def compress(self, series, fillm='bfill'):
        """Return polygonal [length, increment, squared_error] pieces.

        The input is copied and NaNs are filled according to fillm ('bfill'
        by default). With partitions, shared endpoints preserve all intervals.
        """
        
        if self.partition is not None or self.partition_rate is not None:
            series = fillna(series_array(series), fillm)
            pieces = self.parallel_compress(series=series, n_jobs=self.n_jobs)
            pieces = np.vstack(pieces)
            return pieces
        
        return _compress(series=series, tol=self.tol, max_len=self.max_len, fillm=fillm)
    
    
    def parallel_compress(self, series, n_jobs=-1):
        series = fillna(series_array(series), self.fillna)
        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        partition = self.partition
        if partition is None:
            partition = workers
            if self.partition_rate is not None:
                rate = nonnegative(self.partition_rate, "partition_rate")
                if rate == 0:
                    raise ValueError("partition_rate must be positive")
                # Cap before exponentiation to avoid overflow for small rates.
                partition = int(np.exp(min(1 / rate, np.log(len(series))))) * workers
        if isinstance(partition, bool) or not isinstance(partition, (int, np.integer)) or partition < 1:
            raise ValueError("partition must be a positive integer")
        partition = min(partition, len(series) - 1)
        boundaries = np.linspace(0, len(series) - 1, partition + 1, dtype=int)
        chunks = [series[a:b + 1] for a, b in zip(boundaries[:-1], boundaries[1:])]
        self.start_set = [float(chunk[0]) for chunk in chunks]
        with Pool(min(workers, partition)) as pool:
            return pool.starmap(_compress, [(chunk, self.tol, self.max_len) for chunk in chunks])

    @_deprecate_positional_args
    def digitize(self, pieces, alphabet_set=0):
        """Group [length, increment] pieces and return (symbols, Model).

        The functional digitizer and this method share scaling and grouping.
        Centers are in original units. This method alone does not fit the
        estimator or install the returned codebook as self.parameters.
        """

        from .digitization import digitize
        pieces = piece_array(pieces)
        self._std = np.std(pieces, axis=0)
        return digitize(pieces, alpha=self.alpha, sorting=self.sorting,
                        scl=self.scl, alphabet_set=alphabet_set)

    def dump(self, file=None):
        """Save the learned codebook as pickle; prefer JSON via parameters.to_dict()."""
        if not hasattr(self, "parameters"):
            raise NotFittedError("Call fit before exporting parameters.")
        with open("parameters" if file is None else file, "wb") as stream:
            pickle.dump(self.parameters, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file=None, replace=False):
        """Load a trusted pickle. Never load pickle files from untrusted sources."""
        with open("parameters" if file is None else file, "rb") as stream:
            parameters = pickle.load(stream)
        if replace:
            self.parameters = parameters
        else:
            return parameters

    def print_parameters(self):
        """Print the learned symbol-to-center mapping."""
        if not hasattr(self, "parameters"):
            raise NotFittedError("Call fit before inspecting parameters.")
        for symbol, center in zip(self.parameters.alphabets, self.parameters.centers):
            print(f"{symbol}: length={center[0]:g}, increment={center[1]:g}")

    @property
    def tol(self):
        return self._tol
    
    
    
    @tol.setter
    def tol(self, value):
        self._tol = nonnegative(value, "tol")

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
        self._scl = nonnegative(value, "scl")

    @property
    def verbose(self):
        return self._verbose



    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected a float or int type.")
        
        self._verbose = value
        self.logger = logging.getLogger("fABBA")
        


    @property
    def alpha(self):
        return self._alpha
    
    
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = nonnegative(value, "alpha")

    @property
    def max_len(self):
        return self._max_len



    @max_len.setter
    def max_len(self, value):
        self._max_len = max_length(value)

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
    """Return a copy with NaNs filled; boundary gaps use zero for ffill/bfill.

    Methods are zero, mean, median, ffill and bfill (case insensitive).
    Mean/median filling rejects an entirely missing series. Infinity is invalid.
    """
    result = np.array(series, dtype=float, copy=True)
    if result.ndim != 1 or np.isinf(result).any():
        raise ValueError("fillna expects a one-dimensional series without infinity")
    if not isinstance(method, str) or method.lower() not in {'zero', 'mean', 'median', 'ffill', 'bfill'}:
        raise ValueError("fill method must be zero, mean, median, ffill or bfill")
    method = method.lower()
    missing = np.isnan(result)
    if not missing.any():
        return result
    if method in {'mean', 'median'}:
        if missing.all():
            raise ValueError("mean/median filling requires at least one observed sample")
        result[missing] = (np.mean if method == 'mean' else np.median)(result[~missing])
    elif method == 'zero':
        result[missing] = 0
    else:
        indices = range(len(result)) if method == 'ffill' else range(len(result)-1, -1, -1)
        previous = 0.0
        for i in indices:
            if missing[i]:
                result[i] = previous
            else:
                previous = result[i]
    return result
