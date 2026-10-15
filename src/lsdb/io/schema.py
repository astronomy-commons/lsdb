from __future__ import annotations

import pandas as pd
import pyarrow as pa
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN


def get_arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """Constructs the pyarrow schema from a Pandas DataFrame.

    The spatial index column is always placed first, mirroring the column order
    that ``hats-import`` uses when it reduces source shards.

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
    schema = pa.Schema.from_pandas(df).remove_metadata()
    spatial_index_position = schema.get_field_index(SPATIAL_INDEX_COLUMN)
    if spatial_index_position < 1:
        # The column is absent (-1) or already the first field (0).
        return schema
    fields = list(schema)
    fields.insert(0, fields.pop(spatial_index_position))
    return pa.schema(fields)
