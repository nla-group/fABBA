#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Copyright (c) 2021, Stefan Güttel, Xinye Chen
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import os
import pickle
import warnings
import logging
import collections
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import wraps
from inspect import signature, isclass, Parameter
# warnings.filterwarnings('ignore')



try:
    # # %load_ext Cython
    # !python3 setup.py build_ext --inplace
    # from .cagg import aggregate
    from .chainApproximation_c import compress
    from .caggregation_memview import aggregate as aggregate_fc # cython with memory view
except ModuleNotFoundError:
    warnings.warn("cython fail.")
    from .chainApproximation import compress
    from .aggregation import aggregate as aggregate_fc

    
  
  

@dataclass
class Model:
    """
    save ABBA model - parameters
    """
    centers: np.ndarray # store aggregation centers
    splist: np.ndarray # store start point data
    
    """ dictionary """
    hashm: dict # labels -> symbols
    inverse_hashm: dict #  symbols -> labels
    
    




class Aggregation2D:
    """ A separatate aggregation for data with 2-dimensional (2D) features. 
        Independent applicable to 2D data aggregation
        
    Parameters
    ----------
    alpha - float, default=0.5
        Control tolerence for digitization        
    
    sorting - str, default='2-norm', {'lexi', '1-norm', '2-norm'}
        by which the sorting pieces prior to aggregation
    
    Examples
    ----------
    >>> import numpy as np
    >>> from fABBA.symbolic_representation import Aggregation
    >>> n = 50
    >>> x, y = np.random.randn(n), np.random.randn(n)
    >>> x, y = np.concatenate((x, 10*x - 3*y + 40)), np.concatenate((y, y+2*x + 15))
    >>> x, y = x/x.std(), y/y.std()
    >>> data = np.array((x,y)).T
    
    >>> agg = Aggregation2D(sorting='2-norm', alpha=0.5)
    >>> starting_points, labels = agg.aggregate(data)
    >>> print(starting_points, '\n\n', labels)
    [[ 3.80000000e+01  0.00000000e+00 -5.00836436e-03 -1.52431497e-02]
     [ 5.50000000e+01  1.00000000e+00  5.97646389e-01  1.38612117e+00]
     [ 8.40000000e+01  2.00000000e+00  1.20378129e+00  1.60670214e+00]
     [ 8.30000000e+01  3.00000000e+00  1.69650349e+00  1.79137570e+00]
     [ 5.80000000e+01  4.00000000e+00  2.19519185e+00  1.84120254e+00]
     [ 8.70000000e+01  5.00000000e+00  2.15647902e+00  2.45209898e+00]
     [ 9.60000000e+01  6.00000000e+00  2.67780176e+00  2.37982342e+00]] 
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 3 3 3 4 3 1 4 5 4 3 3 3 3 2 2 2 3 3 4 3 4 3 5 1
     2 4 6 3 3 3 4 3 4 3 2 3 3 5 3 2 4 3 3 3 4 3 6 4 4 1]
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

        return np.array(splist), labels

    
    
    
def _deprecate_positional_args(func=None, *, version=None):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    
    from: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/utils/validation.py#L1034
    
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
        strings = fabba.fit_transform(ts)
        fabba.img_norm = (_mean, _std)
    else:
        fabba.img_norm = None
        strings = fabba.fit_transform(ts)
    fabba.img_start = ts[0]
    fabba.img_shape = data.shape
    return strings




def image_decompress(fabba, strings):
    """ image decompression. """
    reconstruction = np.array(fabba.inverse_transform(strings, fabba.img_start))
    if fabba.img_norm != None:
        reconstruction = reconstruction*fabba.img_norm[1] + fabba.img_norm[0]
    reconstruction = reconstruction.round().reshape(fabba.img_shape).astype(np.uint8)
    return  reconstruction



# def load_images(shape=(250, 250)):
#     images = list()
#     folder = os.path.dirname(os.path.realpath(__file__))+'/samples/img'
#     figs = os.listdir(folder)
#     for filename in figs:
#         img = cv2.imread(os.path.join(folder,filename)) 
#         img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB) # transform to grayscale: cv2.COLOR_BGR2GRAY or RGB cv2.COLOR_BGR2RGB
#         img = cv2.resize(img, shape) # resize to 80x80
#         if img is not None:
#             images.append(img)
#     images = np.array(images)
#     return images



class fabba_model(Aggregation2D):
    """
    fABBA: A fast sorting-based aggregation method for symbolic time series representation
    
    Parameters
    ----------
    tol - float, default=0.1
        Control tolerence for compression
    
    alpha - float, default=0.5
        Control tolerence for digitization        
    
    sorting - str, default='2-norm', {'lexi', '1-norm', '2-norm'}
        by which the sorting pieces prior to aggregation
    scl - int, default=1
        Scale for length, default as 1, refers to 2d-digitization, otherwise implement 1d-digitization
    
    verbose - int, default=1
        Verbosity mode, control logs print, default as 1; print logs
    
    max_len - int, default=1
        The max length for each segment, optional choice for compression
    
    string_form - boolean, default=True
        Whether to return with string form
    
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
        * hashmap - dict
            store the oen to one key-value pair for labels earmarked for the groups
            and the corresponding character
    

    
    Examples
    ----------
    >>> import numpy as np
    >>> from fABBA.symbolic_representation import model
    
    >>> np.random.seed(1)
    >>> N = 100
    >>> x = np.random.rand(N)
    
    >>> fabba = model(tol=0.1, alpha=0.5, sorting='lexi', scl=1, verbose=1, max_len=np.inf, string_form=True) 
    >>> print(fabba)
    fABBA(tol=0.1, alpha=0.5, sorting='lexi', scl=1, verbose=1, max_len=inf, string_form=True)
    >>> symbolic_tsf = fabba.fit_transform(x)
    >>> inverse_ts = fabba.inverse_transform(symbolic_tsf, x[0])
    
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, label='time series', c='olive')
    >>> plt.plot(inverse_ts, label='reconstruction', c='darkblue')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()
    
    * In addition to fit_transform, the compression and digitization functions are 
    independent applicable to data. 
    """    
    
    def __init__ (self, tol=0.1, alpha=0.5, 
                  sorting='norm', scl=1, verbose=1,
                  max_len=np.inf, string_form=True, n_jobs=1):
        
        super().__init__()
        self.tol = tol
        self.alpha = alpha
        self.sorting = sorting
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        self.string_form = string_form
        self.n_jobs = n_jobs
        self.compress = compress
        
    
    def __repr__(self):
        parameters_dict = self.__dict__.copy()
        parameters_dict.pop('_std', None)
        parameters_dict.pop('logger', None)
        parameters_dict.pop('parameters', None)
        parameters_dict.pop('compress', None)
        return "%s(%r)" % ("fABBA", parameters_dict)

    
    
    def __str__(self):
        parameters_dict = self.__dict__.copy()
        parameters_dict.pop('_std', None)
        parameters_dict.pop('logger', None)
        parameters_dict.pop('parameters', None)
        parameters_dict.pop('compress', None)
        return "%s(%r)" % ("fABBA", parameters_dict)
    
    
    
    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - numpy.ndarray or list
            Time series of the shape (1, n_samples).
        
        Returns
        ----------
        string (str): The string transformed by fABBA
        """
        
        if self.n_jobs > 1 and self.max_len == 1:
            pieces = self.parallel_compress(ts=series, n_jobs=self.n_jobs)
        else:
            # pieces = self.compress(ts=series)
            pieces = self.compress(ts=series, tol=self.tol, max_len=self.max_len)
            
        string, parameters = self.digitize(
            pieces=np.array(pieces)[:,0:2])
        
        self.parameters = parameters
        
        if self.verbose:
            _info = "Digitization: Reduced pieces of length {}".format(
                len(string)) + " to {} ".format(len(self.parameters.centers)) + " symbols"
            self.logger.info(_info)

        if self.string_form:
            string = "".join(string)
            
        return string

    
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

    

    def parallel_compress(self, ts, n_jobs=-1):
        """
        Approximate a time series using a continuous piecewise linear function in a parallel way.
        Each piece is of length 1. 
        
        Parameters
        ----------
        ts - numpy ndarray
            Time series as input of numpy array

            
        Returns
        -------
        pieces - numpy array
            Numpy ndarray with three columns, each row contains length, increment, error for the segment.
        """
        x = np.arange(0, len(ts))

        def construct_piece(i):
            inc = ts[i+1] - ts[i]
            err = np.linalg.norm((ts[i] + (inc)*x[0:2]) - ts[i:i+2])**2
            return [1, inc, err]

        pieces = Parallel(n_jobs=n_jobs)(
            delayed(construct_piece)(i) for i in range(len(ts) - 1))

        if self.verbose:
            self.logger = logging.getLogger("fABBA")
            self.logger.info(
                "Compression: Reduced time series of length "  
                + str(len(ts)) + " to " + str(len(pieces)) + " segments")

        return np.array(pieces)


    
    @_deprecate_positional_args
    def digitize(self, pieces):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce alpha and len/inc scaling parameter scl.
        A 'temporary' group center, which we call it starting point,
        is used  when assigning pieces to clusters. This temporary
        cluster is the first piece available after appropriate scaling 
        and sorting of all pieces. After finishing the grouping procedure,
        the centers are calculated the mean value of the objects within 
        the clusters
        
        Parameters
        ----------
        pieces - numpy.ndarray
            The compressed pieces of numpy.ndarray with shape (n_samples, n_features) after compression
            
        Returns
        ----------
        string (str or list)
            string sequence
        """

        if self.sorting not in ["lexi", "2-norm", "1-norm", "norm", "pca"]:
            raise ValueError("Please refer to a specific and correct sorting way, namely 'lexi', '2-norm' and '1-norm'")
        
        pieces = np.array(pieces).astype(np.float64)
        self._std = np.std(pieces, axis=0) 
        
        if self._std[0] != 0: # to prevent 0 std when assign max_len as 1 to compression, which make aggregation go wrong.
            npieces = pieces * np.array([self.scl, 1]) / self._std
        else:
            npieces = pieces * np.array([self.scl, 1])
            npieces[:,1] = npieces[:,1] / self._std[1]
        
        if self.sorting in ["lexi", "2-norm", "1-norm"]:
            warnings.warn(f"Pass {self.sorting} as keyword args. From the next version ", 
                          FutureWarning)
            splist, labels = self.aggregate(npieces)
        else:
            labels, splist = aggregate_fc(npieces, self.sorting, self.alpha)
            
        centers = np.zeros((0,2))
        
        for c in range(len(splist)):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
        
        string, hashm, inverse_hashm = self.symbolsAssign(labels)
        
        parameters = Model(centers, np.array(splist), hashm, inverse_hashm)
        return string, parameters



    def symbolsAssign(self, clusters):
        """ automatically assign symbols to different clusters, start with '!'
        Parameters
        ----------
        clusters - list or pd.Series or array
                the list of clusters.
        ----------
        Return:
        
        symbols(list of string), inverse_hash(dict): repectively for the
        corresponding symbolic sequence and the hashmap for inverse transform.
        
        """
        
        clusters = pd.Series(clusters)
        N = len(clusters.unique())

        cluster_sort = [0] * N 
        counter = collections.Counter(clusters)
        for ind, el in enumerate(counter.most_common()):
            cluster_sort[ind] = el[0]

        alphabet= [chr(i) for i in range(33,33 + N)]
        hashm = dict(zip(cluster_sort, alphabet))
        inverse_hashm = dict(zip(alphabet, cluster_sort))
        strings = [hashm[i] for i in clusters]
        return strings, hashm, inverse_hashm



    def inverse_transform(self, strings, start=0, parameters=None):
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
        
        Returns
        -------
        times_series - list
            Reconstruction of the time series.
        """
        if parameters == None:
            pieces = self.inverse_digitize(strings, self.parameters)
        else:
            pieces = self.inverse_digitize(strings, parameters)
            
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(pieces, start)
        return time_series

    
    
    def inverse_digitize(self, strings, parameters):
        """
        Convert symbolic representation back to compressed representation for reconstruction.
        
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
            
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
            
        Returns
        -------
        pieces - np.array
            Time series in compressed format. See compression.
        """
        
        pieces = np.empty([0,2])
        for p in strings:
            pc = parameters.centers[int(parameters.inverse_hashm[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]

    
    
    def quantize(self, pieces):
        """
        Realign window lengths with integer grid.
        
        Parameters
        ----------
        pieces: Time series in compressed representation.
        
        
        Returns
        -------
        pieces: Time series in compressed representation with window length adjusted to integer grid.
        """
            
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
        """
        Reconstruct time series from its first value `ts0` and its `pieces`.
        `pieces` must have (at least) two columns, incremenent and window width, resp.
        A window width w means that the piece ranges from s to s+w.
        In particular, a window width of 1 is allowed.
        
        Parameters
        ----------
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction.
        
        pieces - numpy array
            Numpy array with three columns, each row contains increment, length,
            error for the segment. Only the first two columns are required.
        
        Returns
        -------
        time_series : Reconstructed time series
        """
        
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()

        return time_series
    
    
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
        print("\nHashmap:")
        for i, item in enumerate(cls.parameters.hashm.items()):
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
        if value != np.inf:
            if not isinstance(value, float) and not isinstance(value,int):
                raise TypeError("Expected a float or int type.")
        
        if value <= 0:
            raise ValueError(
                "Please feed an correct value for max_len.")

        self._max_len = value



    @property
    def string_form(self):
        return self._string_form



    @string_form.setter
    def string_form(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean type.")
        self._string_form = value


    @property
    def n_jobs(self):
        return self._n_jobs
    
    
    
    @n_jobs.setter
    def n_jobs(self, value):
        if not isinstance(value, int):
            raise TypeError("Expected a int type.")
        
        self._n_jobs = value

        
        

