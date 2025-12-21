import numpy as np

def compute_storage1(centers, len_strings, num_init=1, bits_for_len=64, bits_for_inc=64, bits_for_ts=64):
    """Compute storage need for ABBA representation"""
    size_centers = centers.shape[0]*bits_for_len + centers.shape[0]*bits_for_inc
    size_strings = 8 * len_strings
    return size_centers + size_strings + bits_for_ts*num_init

def compute_storage2(centers, len_strings, num_init=1, bits_for_len=64, bits_for_inc=64, bits_for_sz=64, bits_for_ts=64):
    """Compute storage need for QABBA representation"""
    # bits_for_sz is for scaling factors
    size_centers = centers.shape[0]*bits_for_len + centers.shape[0]*bits_for_inc
    size_strings = 8 * len_strings
    return size_centers + size_strings + bits_for_ts*num_init + 2*bits_for_sz


def compute_storage(strings, abba, bits_for_len=64, bits_for_inc=64, bits_for_sz=64, bits_for_ts=64):
    centers = abba.parameters.centers
    
    if len(strings[0]) > 1:
        len_strings = np.sum([len(strings[i]) for i in range(len(strings))])
    else:
        len_strings = len(''.join(strings))
        
    if str(type(abba)) == "<class 'fABBA.jabba.jabba.JABBA'>":
        _storage = compute_storage1(centers, len_strings, num_init=abba.new_shape[0], bits_for_len=bits_for_len, 
                                   bits_for_inc=bits_for_inc, bits_for_ts=bits_for_ts)
        
    elif str(type(abba)) == "<class 'fABBA.jabba.qabba.QABBA'>":
        _storage = compute_storage2(centers, len_strings, num_init=abba.new_shape[0], bits_for_len=abba.bits_for_len, 
                                   bits_for_inc=abba.bits_for_inc, bits_for_sz=bits_for_sz, bits_for_ts=bits_for_ts)
    else:
        _storage = compute_storage1(centers, len_strings, num_init=1, bits_for_len=bits_for_len, 
                                   bits_for_inc=bits_for_inc, bits_for_ts=bits_for_ts)
    return _storage
