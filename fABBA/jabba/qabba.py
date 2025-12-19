__all__ = ["quant", "QABBA", "fastQABBA", "fastQABBA_inc", "fastQABBA_len"]

# This software is to simulate quantized ABBA

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import collections
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, Any
import collections
from sklearn.cluster import KMeans
from .fkmns import sampledKMeansInter

from joblib import parallel_backend
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    # # %load_ext Cython
    # !python3 setup.py build_ext --inplace
    from .compmem import compress
    from .aggmem import aggregate # cython with memory view
    from .inversetc import *
except ModuleNotFoundError:
    warnings.warn("cython fail.")
    from .comp import compress
    from .agg import aggregate
    from .inverset import *


import multiprocessing
import subprocess

import warnings 


def check_faiss_installation() -> bool:
    """
    Checks if FAISS is installed and, if so, whether the GPU version is 
    detected by checking the number of available GPUs through the library.

    Returns:
        bool: True if FAISS is installed AND detects one or more GPUs.
    """
    try:
        # Check for FAISS import
        import faiss 
        
        # Check for GPU capability using FAISS's built-in function
        num_gpus = faiss.get_num_gpus()
        
        if num_gpus > 0:
            print(f"[INFO] FAISS installed. GPU version detected with {num_gpus} available GPU(s).")
            return True
        else:
            print("[INFO] FAISS installed (CPU version) or no GPUs detected.")
            return False
            
    except ImportError:
        print("[ERROR] FAISS is NOT installed. Please run 'pip install faiss-cpu' or 'pip install faiss-gpu'.")
        return False


def get_macos_thread_affinity():
    """Attempt to get thread affinity on macOS using Mach thread_policy_get."""
    import ctypes
    from ctypes import c_uint32, c_int, c_size_t
    try:
        libc = ctypes.CDLL('libc.dylib')
        THREAD_AFFINITY_POLICY = 4
        THREAD_AFFINITY_POLICY_COUNT = 1

        class ThreadAffinityPolicyData(ctypes.Structure):
            _fields_ = [("affinity_tag", c_uint32)]

        mach_thread_self = libc.mach_thread_self
        mach_thread_self.restype = c_uint32

        policy_data = ThreadAffinityPolicyData()
        policy_count = c_uint32(THREAD_AFFINITY_POLICY_COUNT)
        get_default = c_int(0)

        ret = libc.thread_policy_get(
            mach_thread_self(),
            c_uint32(THREAD_AFFINITY_POLICY),
            ctypes.byref(policy_data),
            ctypes.byref(policy_count),
            ctypes.byref(get_default)
        )

        if ret == 0 and policy_data.affinity_tag != 0:
            # print(f"macOS thread_policy_get succeeded: Affinity tag {policy_data.affinity_tag}")
            return [policy_data.affinity_tag]
        else:
            print(f"macOS thread_policy_get failed: Error code {ret}, Affinity tag: {policy_data.affinity_tag}")
            return None
        
    except Exception as e:
        print(f"Exception in macOS thread_policy_get: {str(e)}")
        return None

def get_macos_cpu_count():
    """Get the number of available CPUs on macOS using multiprocessing."""
    try:
        cpu_count = multiprocessing.cpu_count()
        # print(f"Retrieved CPU count via multiprocessing: {cpu_count}")
        return list(range(cpu_count))
    except Exception as e:
        print(f"Exception in multiprocessing.cpu_count: {str(e)}")
        return None

def get_macos_cpu_count_sysctl():
    """Get the number of available CPUs on macOS using sysctl."""
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], capture_output=True, text=True, check=True)
        cpu_count = int(result.stdout.strip())
        # print(f"Retrieved CPU count via sysctl: {cpu_count}")
        return list(range(cpu_count))
    except Exception as e:
        print(f"Exception in sysctl hw.ncpu: {str(e)}")
        return None

def get_cpu_affinity():
    """Get CPU affinity for the current process, with enhanced macOS support."""
    import platform
    system = platform.system()
    try:
        import psutil
    except ImportError:
        psutil = None
        
    try:
        # Linux: Try os.sched_getaffinity
        if hasattr(os, 'sched_getaffinity') and system == 'Linux':
            affinity = os.sched_getaffinity(0)
            # print(f"Retrieved Linux affinity: {affinity}")
            return affinity

        # Cross-platform: Try psutil.cpu_affinity
        if psutil is not None:
            process = psutil.Process()
            if hasattr(process, 'cpu_affinity'):
                try:
                    affinity = process.cpu_affinity()
                    # print(f"Retrieved psutil affinity: {affinity}")
                    return affinity
                except Exception as e:
                    print(f"psutil.cpu_affinity failed: {str(e)}")
            else:
                print("psutil.cpu_affinity not available in this version")
               

        # macOS: Try thread affinity and fallbacks
        if system == 'Darwin':
            # print("Attempting macOS thread affinity retrieval...")
            affinity = get_macos_thread_affinity()
            if affinity:
                return affinity
            # print("Falling back to CPU count as macOS affinity proxy...")
            affinity = get_macos_cpu_count()
            if affinity:
                # print("Note: This is the list of available CPUs, not true affinity")
                return affinity
            # print("Falling back to sysctl CPU count...")
            affinity = get_macos_cpu_count_sysctl()
            if affinity:
                # print("Note: This is the list of available CPUs, not true affinity")
                return affinity
            # print("macOS does not provide direct CPU affinity information")

        # Other platforms
        else:
            print(f"CPU affinity not supported on {system}")


    except Exception as e:
        print(f"Unexpected error in get_cpu_affinity: {str(e)}")
        
    return None


class quant():
    """
    Parameters
    ----------
    bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.
        
    sign : bool, default=1
        Whether or not to quantize the value to symmetric integer range.
    
    zpoint : bool, default=1
        Whether or not to compute the zero point. If `zpoint=0`, then the quantized range must be symmetric.
        
    rd_func : function, default=None
        The rounding function used for the quantization. The default is round to nearest.
        
    clip_range : list, default=None
        The clipping function for the quantization.
        
    epsilon : double, default=1e-12
        When the x is comprised of single value, then the scaling factor will be (b - a + epsilon) / (alpha - beta)
        for mapping [a, b] to [alpha, beta].
        
    Methods
    ----------
    quant(x):
        Method that quantize ``x`` to the user-specific arithmetic format.
        
    """
    def __init__(self, bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
        self.bits = bits
        self.sign = sign
        self.zpoint = zpoint
        
        self.rd_func = rd_func
        self.clip_range = clip_range
        self.epsilon = epsilon 
        
        if bits in {8, 16, 32, 64}:
            if bits == 8:
                self.intType = np.int8
                
            elif bits == 16:
                self.intType = np.int16
                
            elif bits == 32:
                self.intType = np.int32
                
            elif bits == 64:
                self.intType = np.int64
                
        else:
            warnings.warn("Current int type not support this bitwidth, use int64 to simulate.")
            self.intType = np.int64
        
        if self.sign == 1:
            if self.zpoint == 1:
                self.alpha_q = -2**(self.bits - 1)
                self.beta_q = 2**(self.bits - 1) - 1
            else:
                self.beta_q = 2**(self.bits - 1) - 1
                self.alpha_q = -self.beta_q
                
        else:
            if self.zpoint == 0:
                self.beta_q = 2**(self.bits - 1) - 1
                self.alpha_q = -self.beta_q
            else:
                raise ValueError('Please set `zpoint` to 0.')
        
        if self.rd_func is None:
            self.rd_func = lambda x: np.round(x, decimals=0)
            
        if self.clip_range is None:
            self.clip_func = lambda x: np.clip(x, a_min=self.alpha_q, a_max=self.beta_q)
        else:
            self.clip_func = lambda x: np.clip(x, a_min=self.clip_range[0], a_max=self.beta_q[self.clip_range[1]])
            
            
    def __call__(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        
        if self.sign != 1:
            abs_max = max(abs(x_min), abs(x_max))
            x_min = -abs_max
            x_max = abs_max
            
        self.scaling, self.zpoint = self.compute_scaling(x_min, x_max, self.alpha_q, self.beta_q)
        
        try:
            x_q = self.quantization(x, self.scaling, self.zpoint)
        except:
            self.scaling, self.zpoint = self.compute_scaling(x_min, x_max+self.epsilon, self.alpha_q, self.beta_q)
            x_q = self.quantization(x, self.scaling, self.zpoint)
            
        return x_q
        
    def dequant(self, x_q):
        return self.dequantization(x_q, self.scaling, self.zpoint)
        
    def quantization(self, x, s, z):
        x_q = self.rd_func((x - z)/s)
        x_q = self.intType(self.clip_func(x_q))
        return x_q

    def dequantization(self, x_q, s, z):
        x_q = x_q.astype(np.float32)
        x = s * x_q + z
        return x

    def compute_scaling(self, alpha, beta, alpha_q, beta_q):
        s = (beta - alpha) / (beta_q - alpha_q)
        z = (alpha * beta_q - beta * alpha_q) / (beta_q - alpha_q)

        return s, z
        
def compute_storage(centers, strings, bits_for_len, bits_for_inc, bits_for_ts=32):
    """Compute storage need for representation"""
    size_centers = centers.shape[0]*bits_for_len + centers.shape[1]*bits_for_inc
    size_strings = 8 * len(strings)
    return size_centers + size_strings + bits_for_ts


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
    
    elif isinstance(alphabet_set, list) and len(alphabets):
        alphabets = alphabet_set
       
    else:
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                    'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z']
        
    clusters = pd.Series(clusters)
    N = len(clusters.unique())

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
    return string.tolist(), alphabets


        
        
@dataclass
class Model:
    """
    save ABBA model - parameters
    """
    centers: np.ndarray # store aggregation centers
    
    """ dictionary """
    alphabets: np.ndarray # labels -> symbols,  symbols -> labels

    

def general_compress(pabba, data, adjust=True, n_jobs=-1):
    if len(data.shape) > 3:
        raise ValueError("Please transform the shape of data into 1D, 2D or 3D.")
    elif len(data.shape) == 3:
        pabba.d_shape = data.shape
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    else:
        pabba.d_shape = data.shape
        
    if adjust:
        _mean = data.mean(axis=0)
        _std = data.std(axis=0)
        if np.any(_std == 0):
            _std[_std == 0] = 1
        data = (data - _mean) / _std
        strings = pabba.fit_transform(data, n_jobs=n_jobs)
        pabba.d_norm = (_mean, _std)
    else:
        pabba.d_norm = None
        strings = pabba.fit_transform(data, n_jobs=n_jobs)

    return strings



def general_decompress(pabba, strings, int_type=True, n_jobs=-1):
    reconstruction = np.array(pabba.inverse_transform(strings, n_jobs=n_jobs))
    if pabba.d_norm is not None:
        try:
            reconstruction = reconstruction*pabba.d_norm[1] + pabba.d_norm[0]
        except ValueError:
            raise ValueError("The number of symbol is not enough to reconstruct this data, please use more symbols.")
            
    if len(pabba.d_shape) == 3:
        if int_type:
            reconstruction = reconstruction.round().reshape(pabba.d_shape).astype(np.uint8)
        else:
            reconstruction = reconstruction.reshape(pabba.d_shape)
            
    return reconstruction



def flatten_to_2d_keep_last(x: Any, keep_last: bool = True) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Flatten all dimensions except the last one (or first one) into a single dimension.
    
    Parameters
    ----------
    x : array-like
        Input of any dimension >= 2
    keep_last : bool
        If True  → keep last dim  → output: (everything_else, last_dim)     ← most common
        If False → keep first dim → output: (first_dim, everything_else)

    Returns
    -------
    x_2d : np.ndarray
        2D array
    original_shape : tuple
        Original shape (for restoring later)
    """
    x = np.asarray(x)                    # handle list, torch, etc.
    
    if x.ndim < 2:
        raise ValueError(f"Input must have at least 2 dimensions, got {x.ndim}")

    original_shape = x.shape

    if keep_last:
        new_shape = (-1, x.shape[-1])    # -1 means "infer this"
        description = f"Flattened all except last dim: {original_shape} -> {new_shape}"
    else:
        new_shape = (x.shape[0], -1)
        description = f"Flattened all except first dim: {original_shape} -> {new_shape}"

    x_2d = x.reshape(new_shape)

    print(description)
    return x_2d, original_shape


def restore_from_2d(x_2d: np.ndarray, original_shape: Tuple[int, ...], keep_last: bool = True) -> np.ndarray:
    """
    Restore the original high-dimensional shape from the 2D flattened version.
    """
    x_restored = x_2d.reshape(original_shape)
    return x_restored


class QABBA(object):
    """
    Simulate QABBA in parallel
    
    Parameters
    ----------        
    tol - double, default=0.5
        Tolerance for compression.
        
    k - int, default=1
        The number of clusters (distinct symbols) specified for ABBA. 
        
    r - float, default=0.5
        The rate of data sampling to perform k-means.  
    
    alpha - double, default=None
        Tolerance for digitization. If None is set, auto-digitization will be enabled.
    
    init - str, default='agg'
        The clustering algorithm in digitization. optional: 'f-kmeans', 'kmeans', 'gpu-kmeans'.
    
    sorting - str, default="norm".
        Apply sorting data before aggregation (inside digitization). Alternative option: "pca".
    
    max_len - int
        The max length of series contained in each compression pieces.
    
    max_iter - int, default=2
        The max iteration for fast k-means algorithm.

    batch_size - int, default=1024
        Size of the mini batches for mini-batch kmeans in digitization. For faster compuations, 
        you can set the batch_size greater than 256 * number of cores to enable parallelism on all cores.

    verbose - int or boolean, default=1
        Enable verbose output.
    
    partition_rate - float or int, default=None
        This parameter is to get the number of partitions of time series. 
        when this parameter is not None, the partitions will be 
        n_jobs*int(np.round(np.exp(1/self.partition_rate), 0))
    
    partition - int:
        The number of subsequences for time series to be partitioned.
         
    scl - int or float, default=1
        Scale the length of compression pieces. The larger the value is, the more important of the length information is.
        Therefore, it can solve some problem resulted from peak shift.
        
    eta - float, default=None,
        Parameter to control the auto-digitization. If None, eta = 3.

    last_dim - boolean, default=True,
        The method to process the varying shape (>=2) of time series. True as default otherwise flatten the shape dimension > 1.
        
    auto_digitize - boolean, default=False
        Enable auto digitization without prior knowledge of alpha.


    Attributes
    ----------
    params: dict
        Parameters of trained model.
    
    string_ - str or list
        Contains the ABBA representation.
    """
    
    def __init__(self, tol=0.2, init='agg', k=2, r=0.5, alpha=None, 
                        sorting="norm", scl=1, max_iter=10,
                        bits_for_len=8, bits_for_inc=16,
                        partition_rate=None, partition=None, 
                        max_len=np.inf, verbose=1, random_state=42, eta=None,
                        fillna='ffill', last_dim=True, auto_digitize=False):
        
        self.tol = tol
        self.alpha = alpha
        self.k = k
        self.scl = scl
        self.init = init
        if self.alpha == None:
            auto_digitize = True
        self.max_len = max_len
        self.max_iter = max_iter
        self.sorting = sorting
        self.verbose = verbose
        self.r = r
        self.eta = eta
        self.partition = partition
        self.partition_rate = partition_rate
        self.temp_symbols = None
        self.symbols_ = None
        self.auto_digitize = auto_digitize
        self.d_norm = None
        self.d_shape = None
        self.eta = None # The coefficient applied to auto digitization. Needless to specify.
        self.fillna = fillna
        self.random_state = random_state
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.recap_shape = None
        self.last_dim = last_dim
        self.stack_last_dim = False
        
    def fit_transform(self, series, n_jobs=-1, alphabet_set=0, return_start_set=False):
        """
        Fitted the numerical series and transform them into symbolic representation.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.

        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        self.fit(series, n_jobs=n_jobs, alphabet_set=alphabet_set)
        if return_start_set:
            return self.string_, self.start_set
        else:
            return self.string_
        
                
    def fit(self, series, n_jobs=-1, alphabet_set=0):
        """
        Fitted the numerical series.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.

        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        self.pieces = self.parallel_compress(series, n_jobs=n_jobs)
        self.string_ = self.digitize(series, self.pieces, alphabet_set, n_jobs)    
        return self
        
    
    
    def parallel_compress(self, series, n_jobs=-1):
        """
        Compress the numerical series in a parallel manner.
        
        Parameters
        ----------        
        series - numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs - int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of available 
            processors the machine allows. Note: For the univariate time series, if n_jobs = 1, 
            PABBA will degenerate to fABBA, but the result may be diferent since PABBA use 
            aggregated groups starting points for reconstruction instead of aggregated groups
            centers.
        """
        

        len_ts = len(series)
        n_jobs = self.n_jobs_init(n_jobs, _max=len_ts)     
        
        if isinstance(series, np.ndarray):
            if len(series.shape) == 1:
                uni_dim = True
            else:
                uni_dim = False
                
        elif isinstance(series, list):
            if not isinstance(series[0], list):
                uni_dim = True
            else:
                uni_dim = False
        else:
            raise ValueError('Please enter time series with correct shape.')
                
        if uni_dim:
            series = np.asarray(series)
            if series.dtype !=  'float64':
                series = np.asarray(series).astype('float64')
                
            # Partition time series for parallelism (for n_jobs > 1 or = -1) if it is univarite
            self.return_series_univariate = True # means the series is univariate,
                                       # so the reconstruction can automatically 
                                       # determine if should return the univariate series.
            if self.partition == None:
                if self.partition_rate == None:
                    partition = n_jobs
                else:
                    partition = int(np.round(np.exp(1/self.partition_rate), 0))*n_jobs
                    if partition > len_ts:
                        warnings.warn("Partition has exceed the maximum length of series.")
                        partition = len_ts
            else:
                if self.partition < len_ts:
                    partition = self.partition
                    if n_jobs > partition: # to prevent useless processors
                        n_jobs = partition
                else:
                    warnings.warn("Partition has exceed the maximum length of series.")
                    partition = n_jobs
                    
            # for i in range(partition,0,-1):
            #    if len_ts % i == 0:
            interval = int(len_ts / partition)
            series = np.vstack([series[i*interval : (i+1)*interval] for i in range(partition)])
                    
            if self.verbose:
                if partition != 1:
                    print("Partition series into {} parts".format(partition))
                print("Init {} processors.".format(n_jobs))
        else:
            self.return_series_univariate = False
            if isinstance(series, np.ndarray):
                if len(series.shape) > 2:
                    if not self.last_dim:
                        self.recap_shape = series.shape
                        series = series.reshape(-1, int(np.prod(self.recap_shape[1:])))
                        
                    else:
                        series, self.recap_shape = flatten_to_2d_keep_last(series)
                    
                    self.stack_last_dim= True

            elif isinstance(series, list):
                pass
            
        pieces = list()
        self.start_set = list()
        
        p = Pool(n_jobs)

        self.start_set = [ts[0] for ts in series]
        pieces = [p.apply_async(compress, args=(fillna(np.asarray(ts).astype(np.double), self.fillna), self.tol, self.max_len)) for ts in series]

        p.close()
        p.join()
        pieces = [p.get() for p in pieces]
        return pieces
    
    
    
    def digitize(self, series, pieces, alphabet_set=0, n_jobs=-1):
        """ Digitization 
        
        Parameters
        ---------- 
        pieces - numpy.ndarray
            The compressed pieces of numpy.ndarray with shape (n_samples, n_features) after compression

        len_ts - int
            The length of time series.
            
        num_pieces - int
            The number of pieces.
            
        init - str
            Use aggregation, fast-kmeans or kmeans for digitization to get symbols.
            
        alphabet_set - int or list, default=0
            The list of alphabet letter. Here provide two different kinds of alphabet letters, namely 0 and 1.
        """
        
        len_ts = len(series)
        
        if len(series.shape) > 1:
            sum_of_length = np.prod(series.shape)
            if self.eta is None: self.eta = 3
        else:
            sum_of_length = len_ts
            if self.eta is None: self.eta = 3
            
            
        num_pieces = list()

        for i in range(len(pieces)):
            num_pieces.append(len(pieces[i]))
        
        pieces = np.vstack(pieces)[:,:2]
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:,0]

        pieces = pieces * np.array([self.scl, 1]) / self._std

        N = pieces[:,:2].shape[0]
        max_k = np.unique(pieces[:,:2],axis=0).shape[0]

        if self.init == 'agg':
            if self.auto_digitize:
                # self.alpha = ( (20 * (sum_of_length - N) * self.tol**2) / (N * (self.eta**4) * sum_of_length**2) ) ** 0.25
                self.alpha = (30 * (sum_of_length - N) * self.tol**2 / (self.eta**4 * sum_of_length))**0.25
                print(f"auto-digitization: alpha={self.alpha}")
            
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            self.k = centers.shape[0]
            
        elif self.init == 'f-kmeans':
            if self.k > max_k:
                self.k = max_k
                warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
            

            with parallel_backend('threading', n_jobs=n_jobs):
                kmeans = sampledKMeansInter(n_clusters=self.k, 
                                            r=self.r, 
                                            init='k-means++', 
                                            max_iter=self.max_iter, 
                                            random_state=self.random_state)
                kmeans.sampled_fit(pieces)
                
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])

        elif self.init == 'gpu-kmeans':
            if check_faiss_installation():
                from .gpu_kmeans import faiss_kmeans_cluster
                centers , labels = faiss_kmeans_cluster(pieces, self.k, self.max_iter)
                centers = centers  * self._std / np.array([self.scl, 1])
                splist = None
            else:
                warnings.warn("PyTorch is not installed or not properly configured for GPU K-means, falling back to CPU K-means.")
                with parallel_backend('threading', n_jobs=n_jobs):
                    kmeans = KMeans(n_clusters=self.k, n_init="auto", random_state=0).fit(pieces)
                    
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
                splist = None

        else: # default => 'kmeans'
            if self.k >= max_k:
                self.k = max_k
                warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
            with parallel_backend('threading', n_jobs=n_jobs):
                kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=1).fit(pieces)
                
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            splist = None
        
        if self.scl == 0:
            centers[:, 0] = one_D_centers(len_pieces, labels, self.k)

        self.quantizer_len = quant(bits=self.bits_for_len)
        self.quantizer_inc = quant(bits=self.bits_for_inc)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        centers[:, 1] = self.quantizer_inc(centers[:, 1])
        
        string, alphabets = symbolsAssign(labels, alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        string_sequences = self.string_separation(string, num_pieces)
        return string_sequences
        
        
        
    def transform(self, series, n_jobs=-1):
        """
        Transform multiple series (numerical sequences) to symbolic sequences.
        
        Parameters
        ----------        
        series: numpy.ndarray, 2-dimension or 1-dimension
            Univariate or multivariate time series
        
        n_jobs: int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of processors 
            the machine allows.  Note: if n_jobs = 1, PABBA will degenerate to fABBA for transfomation.

        """
        
        if series.dtype !=  'float64':
            series = series.astype('float64')
            
        if isinstance(series, np.ndarray):
            if len(series.shape) == 1:
                uni_dim = True
            else:
                uni_dim = False
                
        elif isinstance(series, list):
            if not isinstance(series[0], list):
                uni_dim = True
            else:
                uni_dim = False
        else:
            raise ValueError('Please enter time series with correct shape.')
            
        n_jobs = self.n_jobs_init(n_jobs)
        
        if uni_dim: 
            # Partition time series for parallelism (for n_jobs > 1 or = -1) if it is univarite
            self.return_series_univariate = True # means the series is univariate,
                                       # so the reconstruction can automatically 
                                       # determine if should return the univariate series.
            for i in range(n_jobs,0,-1):
                if series.shape[0] % i == 0:
                    interval = int(series.shape[0] / n_jobs)
                    series = np.vstack([series[i*interval : (i+1)*interval] for i in range(n_jobs)])
        else:
            self.return_series_univariate = False
            
            if self.recap_shape is not None:
                if not isinstance(series, np.ndarray):
                    series = np.asarray(series)

                    if series.shape != self.recap_shape:
                        raise ValueError('Please enter the input with consistent dimensions.')

                series = series.reshape(-1, int(np.prod(self.recap_shape[1:])))
            
        string_sequences = list()
        start_set = list()
        if n_jobs != 1:
            p = Pool(n_jobs)
            for ts in series:
                start_set.append(ts[0])
                string_sequences.append(
                    p.apply_async(self.transform_single_series, args=(ts,))
                )

            p.close()
            p.join()
            string_sequences = [p.get() for p in string_sequences]
        else:
            for ts in series:
                start_set.append(ts[0])
                string_sequences.append(self.transform_single_series(ts,))
        
        return string_sequences, start_set
        
        
        
    def transform_single_series(self, series):
        """
        Transform a single series to symbols.
        
        Parameters
        ----------        
        series: numpy.ndarray, 1-dimension
            Univariate time series
        """
        
        pieces = compress(series, self.tol, self.max_len)
        pieces = np.array(pieces)[:,:2] 
        # pieces = pieces * np.array([self.scl, 1]) / self._std
        symbols_series = list()
        
        for piece_i in pieces:
            symbols_series.append(self.piece_to_symbol(piece_i))
        
        return symbols_series

        
        
    def recast_shape(self, reconstruct_list):
        """Reshape the multiarray to the same shape of the input, the shape might be expanded or squeezed."""
        size_list = [len(i) for i in reconstruct_list]
        fixed_len = self.recap_shape[1] * self.recap_shape[2]
        
        if fixed_len > np.max(size_list):
            warnings.warn('The reconstructed shape has been expanded.', ShapeWarning)
            
        elif fixed_len < np.max(size_list):
            warnings.warn('The reconstructed shape has been squeezed.', ShapeWarning)
        
        org_size = len(reconstruct_list)
        
        if self.recap_shape is not None:
            reconstruct_list.append(fixed_len * [-1])
            pad_token = [np.mean(i) for i in reconstruct_list]
            padded = zip(*zip_longest(*reconstruct_list, fillvalue=pad_token))

            padded = list(padded)
            padded = np.asarray(padded)
            padded = padded[:org_size, :fixed_len].reshape(-1, *self.recap_shape[1:])
            
        else:
            print(f"""Please ensure your fitted series (not this function input) is numpy.ndarray type with dimensions > 2.""")
            
        return padded
    
    
    
    def inverse_transform(self, string_sequences, start_set=None, n_jobs=1):
        """
        Reconstruct the symbolic sequences to numerical sequences.
        
        Parameters
        ----------        
        string_sequences: list
            Univariate or multivariate symbolic time series
        
        start_set: list
            starting value for each symbolic time series reconstruction.
            
        hstack: boolean, default=False
            Determine if concate multiple reconstructed time series into a single time series, 
            which will be useful in the parallelism in univariate time series reconstruction.
            
        n_jobs: int, default=-1
            The mumber of processors used for parallelism. When n_jobs < 0, use all of processors 
            the machine allows.
        """
        
        n_jobs = self.n_jobs_init(n_jobs)
        count = len(string_sequences)
        
        if start_set is None:
            start_set = self.start_set
            if start_set is None:
                raise ValueError('Please input valid start_set.')
        
        inverse_sequences = list()
        
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        if n_jobs != 1 and count != 1:
            p = Pool(n_jobs)
            for i in range(count):
                inverse_sequences.append(
                    p.apply_async(inv_transform,
                                  args=(string_sequences[i],
                                        centers,
                                        self.parameters.alphabets.tolist(),
                                        start_set[i])
                                 )
                )
            p.close()
            p.join()        
            inverse_sequences = [p.get() for p in inverse_sequences]
        else:
            for i in range(count):
                inverse_sequence = inv_transform(string_sequences[i], 
                                                 centers,
                                                 self.parameters.alphabets.tolist(),
                                                 start_set[i]
                                                )
                inverse_sequences.append(inverse_sequence)

        if self.return_series_univariate:
            inverse_sequences = np.hstack(inverse_sequences)
            
        if self.stack_last_dim:
            inverse_sequences = restore_from_2d(np.asarray(inverse_sequences), self.recap_shape)

        return inverse_sequences
        

    
    
    def piece_to_symbol(self, piece):
        """
        Transform a piece to symbol.
        
        Parameters
        ----------        
        piece: numpy.ndarray
            A piece from compression pieces.

        """
        
        
        centers = self.parameters.centers.copy()
        centers[:, 0] = self.quantizer_len.dequant(centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(centers[:, 1])
        
        splabels = np.argmin(np.linalg.norm(centers  - piece, ord=2, axis=1))
        return self.parameters.alphabets[splabels]
    
    
    
    
    def string_separation(self, symbols, num_pieces):
        """
        Separate symbols into symbolic subsequence.
        """
        
        string_sequences = list()
        num_pieces_csum = np.cumsum([0] + num_pieces)
        for index in range(len(num_pieces)):
            string_sequences.append(symbols[num_pieces_csum[index]:num_pieces_csum[index+1]])
        return string_sequences



    def n_jobs_init(self, n_jobs=-1, _max=np.inf):
        """
        Initialize parameter n_jobs.
        """
        
        if n_jobs > _max:
            n_jobs = _max
            warnings.warn("n_jobs init warning, 'n_jobs' is set to maximum {}.".format(n_jobs))
        
        if not isinstance(n_jobs,int):
            raise TypeError('Expected a int type.')
        
        if n_jobs == 0:
            raise ValueError(
                "Please feed an correct value for n_jobs.")
            
        if n_jobs == None or n_jobs == -1:
            cpu_affinity = get_cpu_affinity()
            if cpu_affinity is not None:
                n_jobs = len(cpu_affinity) 
            else:
                n_jobs = 1
            # int(mp.cpu_count()) , return the available usable CPUs
        else:
            n_jobs = n_jobs
        return n_jobs



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
    def alpha(self):
        return self._alpha
    
    
    
    @alpha.setter
    def alpha(self, value):
        if value != None:
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
        if value != np.inf:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected a float or int type.")
        
        if value <= 0:
            raise ValueError(
                "Please feed an correct value for max_len.")

        self._max_len = value
        
        

    @property
    def partition(self):
        return self._partition



    @partition.setter
    def partition(self, value):
        if value != None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected int type.")
        
            if value <= 0:
                raise ValueError(
                    "Please feed an correct value for partition.")

        self._partition = value
        
        
        
    @property
    def partition_rate(self):
        return self._partition_rate



    @partition_rate.setter
    def partition_rate(self, value):
        if value != None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected float or int type.")
        
            if value > 1 or value < 0:
                raise ValueError(
                    "Please feed an correct value for partition_rate.")

        self._partition_rate = value
        
        

    @property
    def scl(self):
        return self._scl



    @scl.setter
    def scl(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("Expected float or int type.")

        if value < 0:
            raise ValueError(
                "Please feed an correct value for scl.")

        self._scl = value
        
        
    @property
    def eta(self):
        return self._eta



    @eta.setter
    def eta(self, value):
        if value is not None:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected float or int type.")

        if value == 0:
            raise ValueError(
                "Please feed an correct value for eta.")

        self._eta = value
        
        
        
    @property
    def k(self):
        return self._k
        
        
        
    @k.setter
    def k(self, value):
        if not isinstance(value,int):
            raise TypeError("Expected int type.")

        if value == 0:
            raise ValueError(
                "Please feed an correct value for k.")

        self._k = value
        
        




class fastQABBA(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.array(series)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.array(series)
                pieces = self._compress(series)
                pieces = np.array(pieces)[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = np.array(pieces)[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_len = quant(bits=self.bits_for_len)
        self.quantizer_inc = quant(bits=self.bits_for_inc)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        centers[:, 1] = self.quantizer_inc(centers[:, 1])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)


class fastQABBA_len(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.array(series)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.array(series)
                pieces = self._compress(series)
                pieces = np.array(pieces)[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = np.array(pieces)[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_len = quant(bits=self.bits_for_len)
        centers[:, 0] = self.quantizer_len(centers[:, 0])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = self.quantizer_len.dequant(self.parameters.centers[:, 0])
        centers[:, 1] = np.float32(self.parameters.centers[:, 1])
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)


class fastQABBA_inc(object): # simulate single time series
    def __init__(self, tol=0.2, k=2, r=1, scl=1, init='agg', 
                 sorting="norm", alpha=0.5,
                 bits_for_len=8, bits_for_inc=16, random_state=2022, n_jobs=1, 
                 alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        
        self.tol = tol
        self.k = k
        self.r = r
        self.scl = scl
        self.max_len = max_len
        self.sorting, self.alpha = sorting, alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.random_state = random_state
        self.verbose = verbose
        self.alphabet_set = alphabet_set
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def transform(self, series):
        series = np.array(series)
        pieces = self._compress(series)
        return self._digitize(pieces)


    def predict(self, series):
        if self.init == 'kmeans':
            if self.kmeans is not None:
                series = np.array(series)
                pieces = self._compress(series)
                pieces = np.array(pieces)[:,:2]
                pieces = pieces * np.array([self.scl, 1]) / self._std

                with parallel_backend('threading', n_jobs=self.n_jobs):
                    labels = self.kmeans.predict(pieces)

                return self.parameters.alphabets[labels]
        
        elif self.init == 'agg':
            raise ValueError("Underdeveloped.")
        else:
            raise ValueError("Please train the model first.")


    def _compress(self, series):
        return compress(series, tol=self.tol, max_len=self.max_len)


    def _digitize(self, pieces):
        pieces = np.array(pieces)[:,:2]
        max_k = np.unique(pieces, axis=0).shape[0]
        if self.k > max_k:
            self.k = max_k
            warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0: # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:
            self._std[1] = 1

        if self.scl == 0:
            len_pieces = pieces[:, 0]
        
        pieces =  pieces * np.array([self.scl, 1]) / self._std
        
        if self.init == 'kmeans':
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.kmeans = sampledKMeansInter(n_clusters=self.k, 
                                                r=self.r, 
                                                init='k-means++', 
                                                max_iter=self.max_iter, 
                                                random_state=self.random_state)

                labels = self.kmeans.sampled_fit_predict(pieces[:,:2])

            if self.scl != 0:
                centers = self.kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])
            else:
                centers = self.kmeans.cluster_centers_ * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
        
        else:
            labels, splist = aggregate(pieces, self.sorting, self.alpha)
            splist = np.array(splist)
            
            if self.scl != 0:
                centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            else:
                centers = splist[:,3:5] * self._std
                centers[:, 0] = one_D_centers(len_pieces, labels, self.k)
            
            self.k = centers.shape[0]
            
        self.quantizer_inc = quant(bits=self.bits_for_inc)
        centers[:, 1] = self.quantizer_inc(centers[:, 1])
        
        self.string_, alphabets = symbolsAssign(labels, self.alphabet_set)
        self.parameters = Model(centers, alphabets)
        
        self.num_grp = self.parameters.centers.shape[0]
        
        if self.verbose:
            print("Generate {} symbols".format(self.num_grp))
        
        return self.string_


    def inverse_transform(self, strings, start):
        centers = np.zeros(self.parameters.centers.shape)
        centers[:, 0] = np.float32(self.parameters.centers[:, 0])
        centers[:, 1] = self.quantizer_inc.dequant(self.parameters.centers[:, 1])
        
        return inv_transform(strings, centers, self.parameters.alphabets.tolist(), start)




def one_D_centers(data, labels, k): # for scl = 0
    centers = np.zeros(k)
    for clust in np.unique(labels):
        centers[clust] = np.mean(data[labels == clust])

    return centers 


# data = np.random.randn(10)
# labels = np.array([1,1,1,0,1,1,1,0,0,0])
# one_D_centers(data, labels)



def dtw(x, y, *, dist=lambda a, b: (a-b)*(a-b), return_path=False, filter_redundant=False):
    """
    Compute dynamic time warping distance between two time series x and y.
    
    Parameters
    ----------
    x - list
        First time series.
    y - list
        Second time series.
    dist - lambda function
        Lambda function defining the distance between two points of the time series.
        By default we use (x-y)^2 to correspond to literature standard for
        dtw. Note final distance d should be square rooted.
    return_path - bool
        Option to return tuple (d, path) where path is a list of tuples outlining
        the route through time series taken to compute dtw distance.
    filter_redundant - bool
        Control filtering to remove `redundant` time series due to sampling
        resolution. For example, if x = [0, 1, 2, 3, 4] and y = [0, 4]. The dynamic
        time series distance is non-zero. If filter_redundant=True then we remove
        the middle 3 time points from x where gradient is constant.
   
    Returns
    -------
    d - numpy float
        Summation of the dist(x[i], y[i]) along the optimal path to minimise overall
        distance. Standard dynamic time warping distance given by default dist and
        d**(0.5).
    
    path - list
        Path taken through time series.
    """

    x = np.array(x)
    y = np.array(y)

    if filter_redundant:
        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x_keep = np.abs(xdiff[1:] - xdiff[0:-1]) >= 1e-14
            x = x[np.hstack((True, x_keep, True))]
        else:
            x_keep = []

        if len(y) > 2:
            ydiff = np.diff(y)
            y_keep = np.abs(ydiff[1:] - ydiff[0:-1]) >= 1e-14
            y = y[np.hstack((True, y_keep, True))]
        else:
            y_keep = []

    len_x, len_y = len(x), len(y)
    window = [(i+1, j+1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float('inf'),))

    if return_path:
        if filter_redundant:
            x_ind = np.arange(1, len(x_keep)+1)
            y_ind = np.arange(1, len(y_keep)+1)
            x_ind = np.hstack((0, x_ind[x_keep], len(x_keep)+1))
            y_ind = np.hstack((0, y_ind[y_keep], len(y_keep)+1))
        else:
            x_ind = np.arange(len(x))
            y_ind = np.arange(len(y))

        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                          (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((x_ind[i-1], y_ind[j-1]))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path)

    else:
        D[0, 0] = 0
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min(D[i-1, j]+dt, D[i, j-1]+dt, D[i-1, j-1]+dt)
        return D[len_x, len_y]
    
    
    
    
    
def fillna(series, method='ffill'):
    """Fill the NA values
    
    Parameters
    ----------   
    series - numpy.ndarray or list
        Time series of the shape (1, n_samples).

    fillna - str, default = 'zero'
        Fill NA/NaN values using the specified method.
        'zero': Fill the holes of series with value of 0.
        'mean': Fill the holes of series with mean value.
        'median': Fill the holes of series with mean value.
        'ffill': Forward last valid observation to fill gap.
            If the first element is nan, then will set it to zero.
        'bfill': Use next valid observation to fill gap. 
            If the last element is nan, then will set it to zero.        
    """

    if method == 'mean':
        series[np.isnan(series)] = np.mean(series[~np.isnan(series)])

    elif method == 'median':
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



class ShapeWarning(UserWarning):
    pass


def zip_longest(*iterables, fillvalue=None):
    # zip_longest('ABCD', 'xy', fillvalue='-') → Ax By C- D-

    iterators = list(map(iter, iterables))
    num_active = len(iterators)
    if not num_active:
        return

    while True:
        values = []
        for i, iterator in enumerate(iterators):
            try:
                value = next(iterator)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = itertools.repeat(fillvalue[i])
                value = fillvalue[i]
            values.append(value)
        yield tuple(values)
