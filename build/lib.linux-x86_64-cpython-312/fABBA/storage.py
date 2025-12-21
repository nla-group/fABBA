import numpy as np


def compute_storage_abba(
    centers,
    len_strings,
    num_init=1,
    bits_for_len=64,
    bits_for_inc=64,
    bits_for_ts=64,
):
    """Compute storage need for ABBA representation"""
    n_centers = centers.shape[0]
    size_centers = n_centers * (bits_for_len + bits_for_inc)
    size_strings = 8 * len_strings  # 1 char = 8 bits
    size_ts = bits_for_ts * num_init
    return size_centers + size_strings + size_ts


def compute_storage_qabba(
    centers,
    len_strings,
    num_init=1,
    bits_for_len=64,
    bits_for_inc=64,
    bits_for_sz=64,
    bits_for_ts=64,
):
    """Compute storage need for QABBA representation"""
    n_centers = centers.shape[0]
    size_centers = n_centers * (bits_for_len + bits_for_inc)
    size_strings = 8 * len_strings
    size_ts = bits_for_ts * num_init
    size_scale = 2 * bits_for_sz  # scaling factors
    return size_centers + size_strings + size_ts + size_scale


def _compute_len_strings(strings):
    """Robust string length computation"""
    if isinstance(strings, str):
        return len(strings)

    if isinstance(strings, (list, tuple)):
        if len(strings) == 0:
            return 0

        # list of lists
        if isinstance(strings[0], (list, tuple)):
            return sum(len(s) for s in strings)

        # list of strings
        return sum(len(s) for s in strings)

    raise TypeError("Unsupported type for strings")


def compute_storage(
    strings,
    abba,
    bits_for_len=64,
    bits_for_inc=64,
    bits_for_sz=64,
    bits_for_ts=64,
):
    centers = abba.parameters.centers
    len_strings = _compute_len_strings(strings)

    # 延迟导入，避免强依赖
    try:
        from fABBA.jabba.jabba import JABBA
        from fABBA.jabba.qabba import QABBA
    except ImportError:
        JABBA = QABBA = ()

    num_init = getattr(abba, "new_shape", [1])[0]

    if isinstance(abba, JABBA):
        storage = compute_storage_abba(
            centers,
            len_strings,
            num_init=num_init,
            bits_for_len=bits_for_len,
            bits_for_inc=bits_for_inc,
            bits_for_ts=bits_for_ts,
        )

    elif isinstance(abba, QABBA):
        storage = compute_storage_qabba(
            centers,
            len_strings,
            num_init=num_init,
            bits_for_len=bits_for_len,
            bits_for_inc=bits_for_inc,
            bits_for_sz=bits_for_sz,
            bits_for_ts=bits_for_ts,
        )

    else:
        # fallback
        storage = compute_storage_abba(
            centers,
            len_strings,
            num_init=1,
            bits_for_len=bits_for_len,
            bits_for_inc=bits_for_inc,
            bits_for_ts=bits_for_ts,
        )

    return storage
