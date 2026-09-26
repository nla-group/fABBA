# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Digitization -- based on aggregation


import numpy as np
from .fabba import Model, symbolsAssign, aggregate_fc, aggregate_fabba
from ._validation import nonnegative, piece_array
from .inverse_t import quantize


def digitize(pieces, alpha=0.5, sorting='norm', scl=1, alphabet_set=0):
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
    
    pieces = piece_array(pieces)
    alpha = nonnegative(alpha, "alpha")
    scl = nonnegative(scl, "scl")
    if sorting not in {"lexi", "2-norm", "1-norm", "norm", "pca"}:
        raise ValueError("sorting must be lexi, 2-norm, 1-norm, norm or pca")
    scale = np.std(pieces, axis=0)
    scale[scale == 0] = 1.0
    npieces = pieces * np.array([scl, 1]) / scale
    # PCA is undefined for a single or entirely constant observation set.
    if sorting == "pca" and (len(pieces) == 1 or np.all(npieces == npieces[0])):
        sorting = "norm"

    if sorting in ["lexi", "2-norm", "1-norm"]:
        # warnings.warn(f"Pass {sorting} as keyword args. From the next version "
        #      f"passing these as positional arguments "
        #      "will result in an error. Additionally, cython implementation will be impossible for this sorting.", 
        #              FutureWarning)
        labels, splist = aggregate_fabba(npieces, sorting, alpha)
    else:
        labels, splist = aggregate_fc(npieces, sorting, alpha)

    centers = np.zeros((0,2))

    for c in range(len(splist)):
        indc = np.argwhere(labels==c)
        center = np.mean(pieces[indc,:], axis=0)
        centers = np.r_[ centers, center ]

    string, alphabets = symbolsAssign(labels, alphabet_set)
    parameters = Model(centers, np.array(splist), alphabets)
    return string, parameters



def inverse_digitize(strings, parameters):
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
    
    from .inverse_t import inv_digitize
    return inv_digitize(strings, parameters.centers, parameters.alphabets.tolist())



def calculate_group_centers(data, labels):
    agg_centers = list() 
    for c in set(labels):
        center = np.mean(data[labels==c,:], axis=0).tolist()
        agg_centers.append( center )
    return np.array(agg_centers)



def wcss(data, labels, centers):
    inertia_ = 0
    for i in np.unique(labels):
        c = centers[i]
        partition = data[labels == i]
        inertia_ = inertia_ + np.sum(np.linalg.norm(partition - c, ord=2, axis=1)**2)
    return inertia_
