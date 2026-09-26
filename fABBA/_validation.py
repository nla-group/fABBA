"""Validation shared by the public compression and digitization APIs."""
from numbers import Real, Integral
import numpy as np


def nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def max_length(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("max_len must be an integer")
    if value != -1 and value < 1:
        raise ValueError("max_len must be -1 (unlimited) or a positive integer")
    return int(value)


def series_array(series):
    raw = np.asarray(series)
    if np.iscomplexobj(raw):
        raise ValueError("series must contain real values")
    array = np.array(raw, dtype=np.float64, copy=True)
    # Preserve the historical row/column-vector convenience, not arbitrary flattening.
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("series must be a one-dimensional sequence of at least two samples")
    if np.isinf(array).any():
        raise ValueError("series must not contain infinity")
    return np.ascontiguousarray(array)


def piece_array(pieces):
    raw = np.asarray(pieces)
    if np.iscomplexobj(raw):
        raise ValueError("pieces must be real")
    array = np.array(raw, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("pieces must be a nonempty matrix with length and increment columns")
    array = array[:, :2].copy()
    if not np.isfinite(array).all() or np.any(array[:, 0] < 1):
        raise ValueError("pieces must be finite with lengths at least one")
    return array
