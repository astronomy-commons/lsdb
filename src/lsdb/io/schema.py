from __future__ import annotations

import dask.dataframe as dd
import pyarrow as pa

from lsdb.operations.operation import Operation


def get_arrow_schema(operation: Operation) -> pa.Schema:
    """Constructs the pyarrow schema from the meta of a Dask DataFrame.

    Parameters
    ----------
    ddf : dd.DataFrame
        A Dask DataFrame.

    Returns
    -------
    pa.Schema
        The arrow schema for the provided Dask DataFrame.
    """
    # pylint: disable=protected-access
    return pa.Schema.from_pandas(operation.meta).remove_metadata()
