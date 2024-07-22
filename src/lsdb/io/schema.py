from __future__ import annotations

from typing import List

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


def filter_schema_by_columns(schema: pa.Schema, columns: List[str] | None = None) -> pa.Schema:
    """Filters the arrow schema according to the columns selected.

    Args:
        schema (pa.Schema): The arrow schema to be filtered.
        columns (List[str]): A list of columns to keep in the schema.

    Returns:
        The arrow schema filtered to contain only the columns specified in `columns`.
    """
    return pa.schema([field for field in schema if field.name in columns]) if columns is not None else schema
