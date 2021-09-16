# Our implemetation of SAX (Symbolic Aggregate approXimation)
## https://cs.gmu.edu/~jessica/SAX_DAMI_preprint.pdf

import string
from scipy.stats import norm
import numpy as np
import warnings

def pca_mean(ts, width):
    if len(ts) % width != 0:
        warnings.warn("Result truncated, width does not divide length")
    return [np.mean(ts[i*width:np.min([len(ts), (i+1)*width])]) for i in range(int(np.floor(len(ts)/width)))]

def reverse_pca(ts, width):
    return np.kron(ts, np.ones([1,width])[0])

def gaussian_breakpoints(ts, symbols):
    # Construct Breakpoints
    breakpoints = np.hstack((norm.ppf([float(a) / symbols for a in range(1, symbols)], scale=1), np.inf))
    ts_GB = ''
    for i in ts:
        for j in range(len(breakpoints)):
            if i < breakpoints[j]:
                ts_GB += chr(97 + j)
                break
    return ts_GB

def reverse_gaussian_breakpoints(ts_GB, number_of_symbols):
    breakpoint_values = norm.ppf([float(a) / (2 * number_of_symbols) for a in range(1, 2 * number_of_symbols, 2)], scale=1)
    ts = []
    for i in ts_GB:
        j = int(ord(i)-97)
        ts.append(breakpoint_values[j])
    return ts


def compress(ts, width = 2):
    reduced_ts = pca_mean(ts, width)
    return reduced_ts

def digitize(ts, number_of_symbols = 5):
    symbolic_ts = gaussian_breakpoints(ts, number_of_symbols)
    return symbolic_ts

def reverse_digitize(symbolic_ts, number_of_symbols = 5):
    reduced_ts = reverse_gaussian_breakpoints(symbolic_ts, number_of_symbols)
    return reduced_ts

def reconstruct(reduced_ts, width = 2):
    ts = reverse_pca(reduced_ts, width)
    return ts

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tslearn.piecewise import SymbolicAggregateApproximation

    # Generate a random walk
    ts = np.random.normal(size = 700)
    ts = np.cumsum(ts)
    ts = ts - np.mean(ts)
    ts /= np.std(ts, ddof=1)

    n_sax_symbols = 8
    n_paa_segments = 10

    # tslearn SAX implementation
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    sax_dataset_inv = sax.inverse_transform(sax.fit_transform(ts))

    # Our SAX implementation
    width = len(ts) // n_paa_segments
    sax = compress(ts, width = width)
    sax = digitize(sax, number_of_symbols = n_sax_symbols)
    sax_ts = reverse_digitize(sax, number_of_symbols = n_sax_symbols)
    sax_ts = reconstruct(sax_ts, width = width)

    plt.figure()
    plt.plot(ts, "b-", alpha=0.4)
    plt.plot(sax_dataset_inv[0].ravel(), "b-")
    plt.plot(sax_ts, 'r--')
    plt.legend(['original', 'tslearn SAX implementation', 'our SAX implementation'])
    plt.title("SAX, %d symbols" % n_sax_symbols)
    plt.show()
