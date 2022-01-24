try:
    from . import chainApproximation_c
    from . import aggregation_memview
    from . import fabba_agg_memview
except ModuleNotFoundError:
    from . import chainApproximation
    from . import digitization
    from . import aggregation

from . import fabba_agg
from . import load_datasets
from . import symbolic_representation
from .symbolic_representation import *
from .chainApproximation import *
from .digitization import *