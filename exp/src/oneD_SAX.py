# Our impementation of 1D-SAX
## https://link.springer.com/chapter/10.1007/978-3-642-41398-8_24

from sklearn import linear_model
from scipy.stats import norm
import numpy as np


def compress(ts, width):
    # See (1) in paper
    pieces = np.empty([0,2]) # gradient, intercept
    for i in range(int(np.floor(len(ts)/width))):
        t = np.arange(i*width, np.min([len(ts), (i+1)*width]))
        Tbar = np.mean(t)
        V = ts[t]
        Vbar = np.mean(V)
        s = np.dot(t-Tbar, V)/np.dot(t-Tbar, t-Tbar)
        b = Vbar - s*Tbar
        a = 0.5*s*(t[0]+t[-1]) + b
        pieces = np.vstack([pieces, np.array([s, a])])
    return pieces


def digitize(pieces, width, slope, intercept):
    # Construct Breakpoints
    breakpoints_slope = np.hstack((norm.ppf([float(a) / slope for a in range(1, slope)], scale = np.sqrt(0.03/width)), np.inf))
    breakpoints_intercept = np.hstack((norm.ppf([float(a) / intercept for a in range(1, intercept)], scale = 1), np.inf))

    symbolic_ts = ''

    for [grad, c] in pieces:
        for i in range(len(breakpoints_slope)):
            if grad < breakpoints_slope[i]:
                cnt1 = i
                break
        for j in range(len(breakpoints_intercept)):
            if c < breakpoints_intercept[j]:
                cnt2 = j
                break
        symbolic_ts += chr((cnt1)*intercept + cnt2 + 97)
    return symbolic_ts


def reverse_digitize(symbolic_ts, width, slope, intercept):

    # Breakpoint midpoints
    slope_values = norm.ppf([float(a) / (2 * slope) for a in range(1, 2 * slope, 2)], scale=np.sqrt(0.03/width))
    intercept_values = norm.ppf([float(a) / (2 * intercept) for a in range(1, 2 * intercept, 2)], scale=1)

    pieces = np.empty([0,2])
    for letter in symbolic_ts:
        j = (ord(letter) - 97) % intercept
        i = (ord(letter) - 97) // intercept
        pieces = np.vstack([pieces,np.array([slope_values[i], intercept_values[j]])])
    return pieces


def reconstruct(pieces, width):
    ts = []
    for s,a in pieces:
        x = (np.array([range(0,width)]))[0]
        x = x - np.mean(x)
        y = s*x + a
        ts = ts + y.tolist()
    return ts


if __name__ == "__main__":
    """
    Compare 1d-SAX implementation against tslearn implementation.
    """
    import matplotlib.pyplot as plt
    from tslearn.piecewise import OneD_SymbolicAggregateApproximation

    # Generate a random walk
    ts = np.random.normal(size = 700)
    ts = np.cumsum(ts)
    ts = ts - np.mean(ts)
    ts /= np.std(ts, ddof=1)
    print('length of ts', len(ts))

    n_sax_symbols_avg = 8
    n_sax_symbols_slope = 8
    n_paa_segments = 10

    # 1d-SAX transform
    one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                    alphabet_size_slope=n_sax_symbols_slope, sigma_l=np.sqrt(0.03/(np.floor(len(ts)/n_paa_segments))))
    one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(ts))

    # Our oneD_SAX
    width = len(ts) // n_paa_segments
    onedsax = compress(ts, width)
    onedsax = digitize(onedsax, width, n_sax_symbols_slope, n_sax_symbols_avg)
    onedsax_ts = reverse_digitize(onedsax, width, n_sax_symbols_slope, n_sax_symbols_avg)
    onedsax_ts = reconstruct(onedsax_ts, width = width)

    # plot
    plt.figure()
    plt.plot(ts, "b-", alpha=0.4)
    plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
    plt.plot(onedsax_ts, 'r--')
    plt.legend(['original', 'tslearn 1D-SAX implementation', 'our 1D-SAX implementation'])
    plt.title("1dSAX, (%d, %d) symbols" % (n_sax_symbols_avg, n_sax_symbols_slope))
    plt.show()
