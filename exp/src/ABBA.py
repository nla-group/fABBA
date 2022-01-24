# Modified from https://github.com/nla-group/ABBA/edit/master/ABBA.py

import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy
import warnings
import collections

class ABBA(object):
    """
    ABBA: Aggregate Brownian bridge-based approximation of time series, see [1].
    Parameters
    ----------
    tol - float/ list
        Tolerance used during compression and digitization. Accepts either float
        or a list of length two. If float given then same tolerance used for both
        compression and digitization. If list given then first element used for
        compression and second element for digitization.
    scl - float
        Scaling parameter in range 0 to infty. Scales the lengths of the compressed
        representation before performing clustering.
    min_k - int
        Minimum value of k, the number of clusters. If min_k is greater than the
        number of pieces being clustered then each piece will belong to its own
        cluster. Warning given.
    max_k - int
        Maximum value of k, the number of clusters.
    max_len - int
        Maximum length of any segment, prevents issue with growing tolerance for
        flat time series.
    verbose - 0, 1 or 2
        Whether to print details.
        0 - Print nothing
        1 - Print key information
        2 - Print all important information
    seed - True/False
        Determine random number generator for centroid initialization during
        sklearn KMeans algorithm. If True, then randomness is deterministic and
        ABBA produces same representation (with fixed parameters) run by run.
    norm - 1 or 2
        Which norm to use for the compression phase. Also used by digitize_inc,
        a greedy clustering approach.
    c_method - 'kmeans' or 'incremental'
        Type of clustering algorithm used
        'kmeans' - Kmeans clustering used, and ckmeans used if scl = 0 or scl = inf
        'incremental' - Cluster increments in a greedy fashion, taking into
            consideration the order of the segments.
    weighted - True/False
        When using c_method = 'incremental, weight elements in clustering due
        to cumulative error.
    Symmetric - True/False
        When using c_method = 'incremental, cluster from both ends to ensure symmetry.


    Raises
    ------
    ValueError: Invalid tol, Invalid scl, Invalid min_k, len(pieces)<min_k.
    Example
    -------
    >>> from ABBA import ABBA
    >>> ts = [-1, 0.1, 1.3, 2, 1.9, 2.4, 1.8, 0.8, -0.5]
    >>> abba = ABBA(tol=0.5, scl=0, min_k=1, max_k = 3)
    >>> string, centers = abba.transform(ts)
    Warning: Time series does not have zero mean.
    Warning: Time series does not have unit variance.
    Compression: Reduced time series of length 9 to 3 segments
    Digitization: Using 2 symbols
    >>> reconstructed_ts = abba.inverse_transform(string, centers, ts[0])
    References
    ------
    [1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based approximation of 		time series. Data Mining and Knowledge Discovery, 34:1175-1200, 2020. 
    """

    def __init__(self, *, tol=0.1, scl=0, min_k=2, max_k=100, max_len = np.inf, verbose=1, seed=True, norm=2, c_method='kmeans', weighted=False, symmetric=True):
        self.tol = tol
        self.scl = scl
        self.min_k = min_k
        self.max_k = max_k
        self.max_len = max_len
        self.verbose = verbose
        self.seed = seed
        self.norm = norm
        self.c_method = c_method
        self.weighted = weighted
        self.symmetric = symmetric

        self._check_parameters()
        
    def _check_time_series(self, time_series):
        # Convert time series to numpy array
        time_series_ = np.array(time_series)

        # Check normalisation if Normalise=False and Verbose
        if self.verbose == 2: # pragma: no cover
            if np.mean(time_series_) > np.finfo(float).eps:
                print('Warning: Time series does not have zero mean.')
            if np.abs(np.std(time_series_) - 1) > np.finfo(float).eps:
                print('Warning: Time series does not have unit variance.')
        return time_series_

    def _check_parameters(self):
        self.compression_tol = None
        self.digitization_tol = None

        # Check tol
        if isinstance(self.tol, list) and len(self.tol) == 2:
            self.compression_tol, self.digitization_tol = self.tol
        elif isinstance(self.tol, list) and len(self.tol) == 1:
            self.compression_tol = self.tol[0]
            self.digitization_tol = self.tol[0]
        elif isinstance(self.tol, float):
            self.compression_tol = self.tol
            self.digitization_tol = self.tol
        else:
            raise ValueError('Invalid tol.')

        # Check scl (scaling parameter)
        if self.scl < 0:
            raise ValueError('Invalid scl.')

        # Check min_k and max_k
        if self.min_k > self.max_k:
            raise ValueError('Invalid limits: min_k must be less than or equal to max_k')

        # Check verbose
        if self.verbose not in [0, 1, 2]: # pragma: no cover
            self.verbose == 1 # set to default
            print('Invalid verbose, setting to default')

        # Check norm
        if self.norm not in [1, 2]:
            raise NotImplementedError('norm = 1 or norm = 2')

        # Check ordered
        if self.c_method not in ['kmeans', 'incremental']:
            raise ValueError('Invalid c_method.')

        # Check weighted
        if type(self.weighted) is not bool:
            raise ValueError('Invalid weighted.')

        # Check symmetric
        if type(self.symmetric) is not bool:
            raise ValueError('Invalid symmetric.')

    def transform(self, time_series):
        """
        Convert time series representation to ABBA symbolic representation
        Parameters
        ----------
        time_series - numpy array
            Normalised time series as numpy array.
        Returns
        -------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            Centres of clusters from clustering algorithm. Each center corresponds
            to character in string.
        """
        time_series_ = self._check_time_series(time_series)
        # Perform compression
        pieces = self.compress(time_series_)

        # Perform digitization
        string, centers = self.digitize(pieces)
        return string, centers

    def inverse_transform(self, string, centers, start=0):
        """
        Convert ABBA symbolic representation back to numeric time series representation.
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            Centers of clusters from clustering algorithm. Each center corresponds
            to character in string.
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction. If not specified, the default is 0.
        Returns
        -------
        times_series - list
            Reconstruction of the time series.
        """

        pieces = self.inverse_digitize(string, centers)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(start, pieces)
        return time_series

    # def compress(self, time_series):
    #     """
    #     Approximate a time series using a continuous piecewise linear function.
    #     Parameters
    #     ----------
    #     time_series - numpy array
    #         Time series as numpy array.
    #     Returns
    #     -------
    #     pieces - numpy array
    #         Numpy array with three columns, each row contains length, increment
    #         error for the segment.
    #     """
    #     start = 0 # start point
    #     end = 1 # end point
    #     pieces = np.empty([0, 3]) # [increment, length, error]
    #     if self.norm == 2:
    #         tol = self.compression_tol**2
    #     else:
    #         tol = self.compression_tol
    #     x = np.arange(0, len(time_series))
    #     epsilon =  np.finfo(float).eps
    # 
    #     (lastinc, lasterr) = (0, 0)
    #     while end < len(time_series):
    #         # error function for linear piece
    #         inc = time_series[end] - time_series[start]
    # 
    #         if self.norm == 2:
    #             err = np.linalg.norm((time_series[start] + (inc/(end-start))*x[0:end-start+1]) - time_series[start:end+1])**2
    #         else:
    #             err = np.linalg.norm((time_series[start] + (inc/(end-start))*x[0:end-start+1]) - time_series[start:end+1],1)
    # 
    #         if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < self.max_len):
    #         # epsilon added to prevent error when err ~ 0 and (end-start-1) = 0
    #             (lastinc, lasterr) = (inc, err)
    #             end += 1
    #             continue
    #         else:
    #             pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    #             start = end - 1
    # 
    #     pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    #     if self.verbose in [1, 2]: # pragma: no cover
    #         print('Compression: Reduced time series of length', len(time_series), 'to', len(pieces), 'segments')
    #     return pieces
    
    
    def compress(self, ts):
        """
        Approximate a time series using a continuous piecewise linear function.

        Parameters
        ----------
        ts - numpy ndarray
            Time series as input of numpy array

        Returns
        -------
        pieces - numpy array
            Numpy ndarray with three columns, each row contains length, increment, error for the segment.
        """

        start = 0
        end = 1
        pieces = list() # np.empty([0, 3])
        x = np.arange(0, len(ts))
        epsilon =  np.finfo(float).eps

        while end < len(ts):
            inc = ts[end] - ts[start]
            err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
            err = np.inner(err, err)

            if (err <= self.tol*(end-start-1) + epsilon) and (end-start-1 < self.max_len):
                (lastinc, lasterr) = (inc, err) 
                end += 1
            else:
                # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
                pieces.append([end-start-1, lastinc, lasterr])
                start = end - 1

        if self.verbose in [1, 2]: # pragma: no cover
            print('Compression: Reduced time series of length', len(time_series), 'to', len(pieces), 'segments')
        pieces.append([end-start-1, lastinc, lasterr])

        return np.array(pieces)
    
    
    def inverse_compress(self, start, pieces):
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

    def _max_cluster_var(self, pieces, labels, centers, k):
        """
        Calculate the maximum variance among all clusters after k-means, in both
        the inc and len dimension.
        Parameters
        ----------
        pieces - numpy array
            One or both columns from compression. See compression.
        labels - list
            List of ints corresponding to cluster labels from k-means.
        centers - numpy array
            centers of clusters from clustering algorithm. Each center corresponds
            to character in string.
        k - int
            Number of clusters. Corresponds to numberof rows in centers, and number
            of unique symbols in labels.
        Returns
        -------
        variance - float
            Largest variance among clusters from k-means.
        """
        d1 = [0] # direction 1
        d2 = [0] # direction 2
        for i in range(k):
            matrix = ((pieces[np.where(labels==i), :] - centers[i])[0]).T
            # Check not all zero
            if not np.all(np.abs(matrix[0,:]) < np.finfo(float).eps):
                # Check more than one value
                if len(matrix[0,:]) > 1:
                    d1.append(np.var(matrix[0,:]))

            # If performing 2-d clustering
            if matrix.shape[0] == 2:
                # Check not all zero
                if not np.all(np.abs(matrix[1,:]) < np.finfo(float).eps):
                    # Check more than one value
                    if len(matrix[1,:]) > 1:
                        d2.append(np.var(matrix[1,:]))
        return np.max(d1), np.max(d2)

    def _build_centers(self, pieces, labels, c1, k, col):
        """
        utility function for digitize, helps build 2d cluster centers after 1d clustering.
        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. See compression.
        labels - list
            List of ints corresponding to cluster labels from k-means.
        c1 - numpy array
            1d cluster centers
        k - int
            Number of clusters
        col - 0 or 1
            Which column was clustered during 1d clustering
        Returns
        -------
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        """
        c2 = []
        for i in range(k):
            location = np.where(labels==i)[0]
            if location.size == 0:
                c2.append(np.NaN)
            else:
                c2.append(np.mean(pieces[location, col]))
        if col == 0:
            return (np.array((c2, c1))).T
        else:
            return (np.array((c1, c2))).T

    def digitize(self, pieces):
        """
        Convert compressed representation to symbolic representation using clustering.
        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. See compression.
        Returns
        -------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        """

        # Check number of pieces
        if len(pieces) < self.min_k:
            raise ValueError('Number of pieces less than min_k.')


        # Construct deep copy and scale data
        data = deepcopy(pieces[:,0:2])

        ########################################################################
        #     'incremental'
        ########################################################################
        if self.c_method == 'incremental':
            labels, centers = self.digitize_incremental(data)

        ########################################################################
        #     'kmeans'
        ########################################################################
        elif self.c_method == 'kmeans':
            if self.scl == np.inf or self.scl == 0:
                labels, centers = self.digitize_ckmeans(data)
            else:
                labels, centers = self.digitize_kmeans(data)

        ########################################################################
        #     Convert labels
        ########################################################################
        # Order cluster centres so 'a' is the most populated cluster, and so forth.
        k = len(set(labels))
        new_to_old = [0] * k
        counter = collections.Counter(labels)
        for ind, el in enumerate(counter.most_common()):
            new_to_old[ind] = el[0]

        # invert permutation
        old_to_new = [0] * k
        for i, p in enumerate(new_to_old):
            old_to_new[p] = i

        # Convert labels to string
        string = ''.join([ chr(97 + old_to_new[j]) for j in labels ])
        return string, centers[new_to_old, :]

    def digitize_ckmeans(self, data):
        # Initialise variables
        centers = np.zeros((0,2))
        labels = [-1]*np.shape(data)[0]
        # Try Cpp wrapper
        try:
            from src.Ckmeans import kmeans_1d_dp
            from src.Ckmeans import double_vector
            self.Ck = True
        except: #TODO use https://github.com/llimllib/ckmeans/blob/master/ckmeans.py instead
            self.Ck = False
            if self.verbose in [1, 2]: # pragma: no cover
                warnings.warn('Ckmeans module unavailable, try running makefile. Using sklearn KMeans instead.',  stacklevel=3)

        ########################################################################
        #     scl == 0
        ########################################################################
        if self.scl == 0:
            # construct tol_s
            s = .20
            N = 1
            for i in data:
                N += i[0]
            bound = ((6*(N-len(data)))/(N*len(data)))*((self.digitization_tol*self.digitization_tol)/(s*s))

            # scale inc to unit variance
            inc_std = np.std(data[:,1])
            inc_std = inc_std if inc_std > np.finfo(float).eps else 1
            data[:,1] /= inc_std

            # Check if CKmeans compatible
            if self.Ck and (len(set(data[:,1])) < self.min_k):
                if self.verbose in [1, 2]: # pragma: no cover
                    warnings.warn('Note enough unique pieces for Ckmeans. Using sklearn KMeans instead.',  stacklevel=3)
                self.Ck = False

            # Use C++ CKmeans
            if self.Ck: # pragma: no cover
                d = double_vector(data[:,1])
                output = kmeans_1d_dp(d, self.min_k, self.max_k, bound, 'linear')
                labels = np.array(output.cluster)

                c = np.array(output.centres)
                c *= inc_std
                centers = self._build_centers(data, labels, c, output.Kopt, 0)

                if self.verbose in [1, 2]: # pragma: no cover
                    print('Digitization: Using', output.Kopt, 'symbols')

                k = output.Kopt

            # Use Kmeans
            else:
                # Run through values of k from min_k to max_k checking bound
                if self.digitization_tol != 0:
                    error = np.inf
                    k = self.min_k - 1
                    while k <= self.max_k-1 and (error > bound):
                        k += 1
                        # tol=0 ensures labels and centres coincide
                        if self.seed:
                            kmeans = KMeans(n_clusters=k, tol=0, random_state=0).fit(data[:,1].reshape(-1,1))
                        else:
                            kmeans = KMeans(n_clusters=k, tol=0).fit(data[:,1].reshape(-1,1))
                        centers = kmeans.cluster_centers_
                        labels = kmeans.labels_

                        error_1, error_2 = self._max_cluster_var(data[:,1].reshape(-1,1), labels, centers, k)
                        error = max([error_1, error_2])
                        if self.verbose == 2: # pragma: no cover
                            print('k:', k)
                            print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)
                    if self.verbose in [1, 2]: # pragma: no cover
                        print('Digitization: Using', k, 'symbols')

                # Zero error so cluster with largest possible k.
                else:
                    if len(data[:,1]) < self.max_k:
                        k = len(data[:,1])
                    else:
                        k = self.max_k

                    # tol=0 ensures labels and centres coincide
                    kmeans = KMeans(n_clusters=k, tol=0).fit(data[:,1].reshape(-1,1))
                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    error = self._max_cluster_var(data[:,1].reshape(-1,1), labels, centers, k)
                    if self.verbose in [1, 2]: # pragma: no cover
                        print('Digitization: Using', k, 'symbols')

                # build cluster centers
                c = centers.reshape(1,-1)[0]
                c *= inc_std
                centers = self._build_centers(data, labels, c, k, 0)

        ########################################################################
        #     scl == inf
        ########################################################################
        elif self.scl == np.inf:
            # construct tol_s
            s = .20
            N = 1
            for i in data:
                N += i[0]
            bound = ((6*(N-len(data)))/(N*len(data)))*((self.digitization_tol*self.digitization_tol)/(s*s))

            # scale length to unit variance
            len_std = np.std(data[:,0])
            len_std = len_std if len_std > np.finfo(float).eps else 1
            data[:,0] /= len_std

            # Select first column and check unique for Ckmeans
            if self.Ck and (len(set(data[:,0])) < self.min_k):
                if self.verbose in [1, 2]: # pragma: no cover
                    warnings.warn('Note enough unique pieces for Ckmeans. Using sklearn KMeans instead.',  stacklevel=3)
                self.Ck = False

            # Use Ckmeans
            if self.Ck: # pragma: no cover
                d = double_vector(data[:,0])
                output = kmeans_1d_dp(d, self.min_k, self.max_k, bound, 'linear')
                labels = np.array(output.cluster)

                c = np.array(output.centres)
                c *= len_std
                centers = self._build_centers(data, labels, c, output.Kopt, 1)

                if self.verbose in [1, 2]: # pragma: no cover
                    print('Digitization: Using', output.Kopt, 'symbols')

                k = output.Kopt

            # Use Kmeans
            else:
                # Run through values of k from min_k to max_k checking bound
                if self.digitization_tol != 0:
                    error = np.inf
                    k = self.min_k - 1
                    while k <= self.max_k-1 and (error > bound):
                        k += 1
                        # tol=0 ensures labels and centres coincide
                        if self.seed:
                            kmeans = KMeans(n_clusters=k, tol=0, random_state=0).fit(data[:,0].reshape(-1,1))
                        else:
                            kmeans = KMeans(n_clusters=k, tol=0).fit(data[:,0].reshape(-1,1))
                        centers = kmeans.cluster_centers_
                        labels = kmeans.labels_
                        error_1, error_2 = self._max_cluster_var(data[:,0].reshape(-1,1), labels, centers, k)
                        error = max([error_1, error_2])
                        if self.verbose == 2: # pragma: no cover
                            print('k:', k)
                            print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)
                    if self.verbose in [1, 2]: # pragma: no cover
                        print('Digitization: Using', k, 'symbols')

                # Zero error so cluster with largest possible k.
                else:
                    if len(data[:,0]) < self.max_k:
                        k = len(data[:,0])
                    else:
                        k = self.max_k

                    # tol=0 ensures labels and centres coincide
                    kmeans = KMeans(n_clusters=k, tol=0).fit(data[:,0].reshape(-1,1))
                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    error = self._max_cluster_var(data[:,0].reshape(-1,1), labels, centers, k)
                    if self.verbose in [1, 2]: # pragma: no cover
                        print('Digitization: Using', k, 'symbols')

                # build cluster centers
                c = centers.reshape(1,-1)[0]
                c *= len_std
                centers = self._build_centers(data, labels, c, k, 1)
        return labels, centers

    def digitize_kmeans(self, data):
        # Initialise variables
        centers = np.zeros((0,2))
        labels = [-1]*np.shape(data)[0]
        ########################################################################
        #     scl in (0, inf)
        ########################################################################
        # construct tol_s
        s = .20
        N = 1
        for i in data:
            N += i[0]
        bound = ((6*(N-len(data)))/(N*len(data)))*((self.digitization_tol*self.digitization_tol)/(s*s))

        # scale length to unit variance
        len_std = np.std(data[:,0])
        len_std = len_std if len_std > np.finfo(float).eps else 1
        data[:,0] /= len_std

        # scale inc to unit variance
        inc_std = np.std(data[:,1])
        inc_std = inc_std if inc_std > np.finfo(float).eps else 1
        data[:,1] /= inc_std

        # Kmeans
        data[:,0] *= self.scl # scale lengths accordingly
        # Run through values of k from min_k to max_k checking bound
        if self.digitization_tol != 0:
            error = np.inf
            k = self.min_k - 1
            while k <= self.max_k-1 and (error > bound):
                k += 1
                # tol=0 ensures labels and centres coincide
                if self.seed:
                    kmeans = KMeans(n_clusters=k, tol=0, random_state=0).fit(data)
                else:
                    kmeans = KMeans(n_clusters=k, tol=0).fit(data)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                error_1, error_2 = self._max_cluster_var(data, labels, centers, k)
                error = max([error_1, error_2])
                if self.verbose == 2: # pragma: no cover
                    print('k:', k)
                    print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)
            if self.verbose in [1, 2]: # pragma: no cover
                print('Digitization: Using', k, 'symbols')

        # Zero error so cluster with largest possible k.
        else:
            if len(data) < self.max_k:
                k = len(data)
            else:
                k = self.max_k

            # tol=0 ensures labels and centres coincide
            kmeans = KMeans(n_clusters=k, tol=0).fit(data)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            error = self._max_cluster_var(data, labels, centers, k)
            if self.verbose in [1, 2]: # pragma: no cover
                print('Digitization: Using', k, 'symbols')

        # build cluster centers
        c = centers.reshape(1,-1)[0]
        centers[:,0] *= len_std
        centers[:,0] /= self.scl # reverse scaling
        centers[:,1] *= inc_std
        return labels, centers

    def digitize_incremental(self, data):
        """
        Convert compressed representation to symbolic representation using 1D clustering.
        This method clusters only the increments of the pieces and is greedy.
        It is tolerance driven.
        """

        def weighted_median(data, weights):
            """
            Args:
              data (list or numpy.array): data
              weights (list or numpy.array): weights
            Taken from https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
            """
            data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
            s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
            midpoint = 0.5 * sum(s_weights)
            if any(weights > midpoint):
                w_median = (data[weights == np.max(weights)])[0]
            else:
                cs_weights = np.cumsum(s_weights)
                idx = np.where(cs_weights <= midpoint)[0][-1]
                if cs_weights[idx] == midpoint:
                    w_median = np.mean(s_data[idx:idx+2])
                else:
                    w_median = s_data[idx+1]
            return w_median

        # Initialise variables
        centers = np.zeros((0,2))
        labels = [-1]*np.shape(data)[0]

        if self.symmetric:
            ind = np.argsort(abs(data[:,1]))
        else:
            ind = np.argsort(data[:,1])

        k = 0 # counter for clusters
        inds = 0    # given accepted cluster
        inde = 0
        mval = data[ind[inds], 1]

        last_sign = np.sign(mval)  # as soon as there is a cluster having a sign change in increments
        sign_change = False        # we have covered the point zero. from that on we should work
        sign_sorted = False        # incrementally in the positive and negative direction

        while inde < np.shape(data)[0]:
            if inde == np.shape(data)[0]-1:
                #print('final')
                old_mval = mval
                nrmerr = np.inf
            else:
                # try to add another point to cluster
                vals = data[np.sort(ind[inds:inde+2]), 1]

                if np.sign(data[ind[inde+1], 1]) != last_sign: # added point has different sign
                    sign_change = True

                ell = inde-inds+2 # number of points in new test cluster
                old_mval = mval

                if self.weighted and self.norm==1: # minimize accumulated increment errors in 1-norm
                    wgts = np.arange(1,ell+1)
                    wvals = np.cumsum(vals)/wgts
                    mval = weighted_median(wvals, wgts)
                    err = np.cumsum(vals) - np.arange(1,ell+1)*mval
                    nrmerr = np.linalg.norm(err,1)

                if self.weighted and self.norm==2: # minimize accumulated increment errors in 2-norm
                    wgths = (ell+1)*ell/2 - np.cumsum(np.arange(0,ell))
                    wvals = vals*wgths
                    mval = np.sum(wvals)/((ell)*(ell+1)*(2*ell+1)/6)
                    err = np.cumsum(vals) - np.arange(1,ell+1)*mval
                    nrmerr = np.linalg.norm(err)**2

                if not self.weighted and self.norm==1: # minimize nonaccumulated increment errors in 1-norm
                    mval = np.median(vals)   # standard median
                    err = vals - np.ones((1,ell))*mval
                    nrmerr = np.linalg.norm(err,1)

                if not self.weighted and self.norm==2: # minimize nonaccumulated increment errors in 2-norm
                    mval = np.sum(vals)/ell  # standard mean
                    err = vals - np.ones((1,ell))*mval
                    nrmerr = np.linalg.norm(err)**2

            if nrmerr < ell*self.digitization_tol and inde+1<np.shape(data)[0]:   # accept enlarged cluster
                inde += 1
            else:
                mlen = np.mean(data[ind[inds:inde+1], 0])
                for ii in ind[inds:inde+1]:
                    labels[ii] = k
                centers = np.vstack((centers, np.array([mlen, old_mval])))

                if self.symmetric and not sign_sorted and sign_change:
                    ind1 = ind[inde+1:]
                    lst = data[ind1, 1]
                    ind2 = np.lexsort((np.abs(lst),np.sign(lst)))
                    ind[inde+1:] = ind1[ind2]
                    sign_sorted = True

                k += 1
                inds = inde+1
                inde = inds

                if inds < np.shape(data)[0]:
                    mval = data[ind[inds], 1]
        return labels, centers

    def inverse_digitize(self, string, centers):
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
        for p in string:
            pc = centers[ord(p)-97,:]
            pieces = np.vstack([pieces, pc])
        return pieces

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
            pieces[-1,0] = round(pieces[-1,0])
        return pieces

    def get_patches(self, ts, pieces, string, centers):
        """
        Creates a dictionary of patches from time series data using the clustering result.
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
            centers of clusters from clustering algorithm. Each centre corresponds
            to a character in string.

        Returns
        -------
        patches - dict
            A dictionary of time series patches.
        """

        patches = dict()
        inds = 0
        for j in range(len(pieces)):
            let = string[j]                           # letter
            lab = ord(string[j])-97                   # label (integer)
            lgt = round(centers[lab,0])               # patch length
            inc = centers[lab,1]                      # patch increment
            inde = inds + int(pieces[j,0]);
            tsp = ts[inds:inde+1]                      # time series patch

            tsp = tsp - (tsp[-1]-tsp[0]-inc)/2-tsp[0]  # shift patch so that it is vertically centered with patch increment

            tspi = np.interp(np.linspace(0,1,lgt+1), np.linspace(0,1,len(tsp)), tsp)
            if let in patches:
                patches[let] = np.append(patches[let], np.array([tspi]), axis = 0)
            else:
                patches[let] = np.array([ tspi ])
            inds = inde
        return patches

    def patched_reconstruction(self, time_series, pieces, string, centers):
        """
        An alternative reconstruction procedure which builds patches for each
        cluster by extrapolating/intepolating the segments and taking the mean.
        The reconstructed time series is no longer guaranteed to be of the same
        length as the original.
        Parameters
        ----------
        time_series - numpy array
            Normalised time series as numpy array.
        pieces - numpy array
            One or both columns from compression. See compression.
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            centers of clusters from clustering algorithm. Each center corresponds
            to character in string.

        """
        patches = self.get_patches(time_series, pieces, string, centers)
        # Construct mean of each patch
        d = {}
        for key in patches:
            d[key] = list(np.mean(patches[key], axis=0))

        reconstructed_time_series = [time_series[0]]
        for letter in string:
            patch = d[letter]
            patch -= patch[0] - reconstructed_time_series[-1] # shift vertically
            reconstructed_time_series = reconstructed_time_series + patch[1:].tolist()
        return reconstructed_time_series

    def plot_patches(self, patches, string, centers, ts0=0, xoffset=0): # pragma: no cover
        """
        Plot stitched patches.
        Parameters
        ----------
        patches - dict
            Dictionary of patches as returned by get_patches.

        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.

        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to a character in string.

        ts0 - float
            First time series value (default 0).

        xoffset - float
            Start index on x-axis for plotting (default 0)
        """
        import matplotlib.pyplot as plt
        inds = xoffset
        val = ts0
        for j in range(len(string)):
            let = string[j]                           # letter
            lab = ord(string[j])-97                   # label (integer)
            lgt = int(centers[lab,0])               # patch length
            inc = centers[lab,1]                      # patch increment
            inde = inds + lgt
            xp = np.arange(inds,inde+1,1)             # time series x-vals
            plt.plot(xp,patches[let].T+val,'k-',color=(0.8, 0.8, 0.8));
            val = val + inc
            inds = inde

        # now plot solid polygon on top
        inds = xoffset
        val = ts0
        for j in range(len(string)):
            let = string[j]                           # letter
            lab = ord(string[j])-97                   # label (integer)
            lgt = round(centers[lab,0])               # patch length
            inc = centers[lab,1]                      # patch increment
            inde = inds + lgt
            xp = np.arange(inds,inde+1,1)             # time series x-vals
            plt.plot([inds,inde],[val,val+inc],'b-')
            val = val + inc
            inds = inde
