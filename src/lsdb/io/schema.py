from __future__ import annotations

import dask.dataframe as dd
import pyarrow as pa


def get_arrow_schema(ddf: dd.DataFrame) -> pa.Schema:
    """Constructs the pyarrow schema from the meta of a Dask DataFrame.

    Args:
        ddf (dd.DataFrame): A Dask DataFrame.

    Returns:
        The arrow schema for the provided Dask DataFrame.
    """
    # pylint: disable=protected-access
    return pa.Schema.from_pandas(ddf._meta)
