try:
    import scipy
    if scipy.__version__ != '1.8.0':
        from .separate.aggregation_cm import aggregate
    else:
        from .separate.aggregation_c import aggregate
    from .extmod.inverse_tc import *
    
    
except (ModuleNotFoundError, ValueError):
    from .separate.aggregation import aggregate
    
    
__version__ = '1.0.7'
from .load_datasets import load_images
from .symbolic_representation import (image_compress, image_decompress, ABBAbase, ABBA,
                                      get_patches, patched_reconstruction, fABBA,
                                      symbolsAssign, fillna)

from .symbolic_representation import _compress as compress
from .symbolic_representation import _inverse_compress as inverse_compress
from .digitization import digitize, inverse_digitize, quantize, wcss, calculate_group_centers
