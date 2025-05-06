import nested_pandas as npd
import pandas as pd
from dask._dispatch import get_collection_type
from dask.dataframe.backends import meta_nonempty_dataframe
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.extensions import make_array_nonempty
from dask.dataframe.utils import meta_nonempty
from nested_pandas.series.ext_array import NestedExtensionArray

from .core import NestedFrame

get_collection_type.register(npd.NestedFrame, lambda _: NestedFrame)

# The following dispatch functions are defined as per the Dask extension guide:
# https://docs.dask.org/en/latest/dataframe-extend.html


@make_meta_dispatch.register(npd.NestedFrame)
def make_meta_frame(x, index=None) -> npd.NestedFrame:
    """Create an empty NestedFrame to use as Dask's underlying object meta."""

    dtypes = x.dtypes.to_dict()
    index = index if index is not None else x.index
    index = index[:0].copy()
    result = npd.NestedFrame({key: pd.Series(dtype=d) for key, d in dtypes.items()}, index=index)
    return result


@meta_nonempty.register(npd.NestedFrame)
def _nonempty_nestedframe(x, index=None) -> npd.NestedFrame:
    """Construct a new NestedFrame with the same underlying data."""
    df = meta_nonempty_dataframe(x)
    if index is not None:
        df.index = index
    return npd.NestedFrame(df)


@make_array_nonempty.register(npd.NestedDtype)
def _(dtype) -> NestedExtensionArray:
    """Register a valid dtype for the NestedExtensionArray"""
    # must be two values to avoid a length error in meta inference
    # Dask seems to explicitly require meta dtypes to have length 2.
    vals = [pd.NA, pd.NA]
    return NestedExtensionArray._from_sequence(vals, dtype=dtype)  # pylint: disable=protected-access
