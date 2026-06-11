from __future__ import annotations

import pandas as pd
import pyarrow as pa


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
