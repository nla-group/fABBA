"""Validated reconstruction shared by the public univariate APIs."""
import numpy as np
from ._validation import piece_array


def inv_transform(strings, centers, alphabets, start=0):
    """Decode symbols, realign lengths to the integer grid, and reconstruct."""
    pieces = inv_digitize(strings, centers, alphabets)
    return inv_compress(quantize(pieces), start)


def inv_digitize(strings, centers, alphabets):
    """Return independent [length, increment] rows for a symbol sequence."""
    centers = piece_array(centers)
    if len(alphabets) != len(centers) or len(set(alphabets)) != len(alphabets):
        raise ValueError("alphabets must have one unique symbol per center")
    lookup = dict(zip(alphabets, range(len(alphabets))))
    try:
        indices = [lookup[symbol] for symbol in strings]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown or invalid symbol: {exc}") from exc
    return centers[indices].copy()


def quantize(pieces):
    """Round cumulative lengths without mutating input; preserve total duration.

    Ties round upward, so segments of length one cannot collapse at half-grid
    boundaries. Every decoded segment has at least one interval. Empty sequences
    are valid.
    """
    if np.asarray(pieces).size == 0:
        return np.empty((0, 2), dtype=float)
    result = piece_array(pieces)
    ends = np.floor(np.cumsum(result[:, 0]) + 0.5)
    result[:, 0] = np.diff(np.r_[0, ends])
    return result


def inv_compress(pieces, start):
    """Reconstruct from [length, increment] rows and a finite starting value.

    Lengths must be positive integers; use quantize for learned fractional lengths.
    """
    if not np.isscalar(start) or not np.isreal(start) or not np.isfinite(start):
        raise ValueError("start must be a finite real scalar")
    if np.asarray(pieces).size == 0:
        return [float(start)]
    pieces = piece_array(pieces)
    if np.any(pieces[:, 0] != np.rint(pieces[:, 0])):
        raise ValueError("segment lengths must be integers; call quantize first")
    chunks = [np.array([float(start)])]
    endpoint = float(start)
    for length, increment in pieces:
        chunks.append(endpoint + np.arange(1, int(length) + 1) * (increment / length))
        endpoint += increment
    return np.concatenate(chunks).tolist()
