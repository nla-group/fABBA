try:
    import numpy, scipy
    if scipy.__version__ != '1.8.0':
        from .separate.aggregation_cm import *
    else:
        from .separate.aggregation_c import *
        
    if numpy.__version__ >= '1.22.0':
        from .extmod.chainApproximation_cm import *
    else:
        from .separate.aggregation_c import *
        from .extmod.chainApproximation_c import *
        
except (ModuleNotFoundError, ValueError):
    from .chainApproximation import *
    from .separate.aggregation import *

from .load_datasets import load_images
from .symbolic_representation import *
from .digitization import *