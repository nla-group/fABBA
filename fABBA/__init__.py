try:
    try:
        from .separate.aggregation_cm import aggregate
    except:
        from .separate.aggregation_c import aggregate
    
    from .extmod.inverse_tc import *
    
except (ModuleNotFoundError, ValueError):
    from .separate.aggregation import aggregate
    
    
__version__ = '1.4.3'
from .load_datasets import load_images, loadData
from .fabba import (image_compress, image_decompress, ABBAbase, ABBA,
                                      get_patches, patched_reconstruction, fABBA,
                                      symbolsAssign, fillna)


from .fabba import Model
from .fabba import _compress as compress
from .fabba import _inverse_compress as inverse_compress
from .digitization import digitize, inverse_digitize, quantize, wcss, calculate_group_centers

from .jabba.jabba import fastABBA, JABBA
from .jabba import jabba
from .jabba import preprocessing
from .jabba.fkmns import sampledKMeansInter as fkmeans

from .jabba import qabba
from .jabba.qabba import *

from .jabba import xabba
from .jabba.xabba import XABBA

