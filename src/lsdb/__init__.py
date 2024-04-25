import warnings

import dask

QUERY_PLANNING_ON = dask.config.get("dataframe.query-planning")
# Force the use of dask-expressions backends
if QUERY_PLANNING_ON is not False:
    warnings.warn("This version of lsdb does not support dataframe query-planning, which has been disabled.")
    dask.config.set({"dataframe.query-planning": False})

# pylint: disable=wrong-import-position
from ._version import __version__
from .catalog import Catalog
from .loaders import from_dataframe, read_hipscat
