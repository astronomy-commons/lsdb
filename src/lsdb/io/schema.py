from __future__ import annotations

import pyarrow as pa
import pandas as pd

from lsdb.operations.operation import Operation


def get_arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """Constructs the pyarrow schema from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame to extract the schema from.

    Returns
    -------
    pa.Schema
        The arrow schema for the provided DataFrame.
    """
    # pylint: disable=protected-access
    return pa.Schema.from_pandas(df).remove_metadata()
