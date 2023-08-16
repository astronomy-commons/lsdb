import pandas as pd
from hipscat.io import FilePointer
from pyarrow import Schema
import pyarrow.parquet as pq


def read_parquet_schema(file_pointer: FilePointer) -> Schema:
    """Reads the schema from a parquet file

    Args:
        file_pointer (FilePointer): File Pointer to a parquet file

    Returns:
        PyArrow schema object with the schema of the parquet file
    """
    return pq.read_schema(file_pointer)


def read_parquet_file_to_pandas(file_pointer: FilePointer, **kwargs) -> pd.DataFrame:
    """Reads a parquet file to a pandas DataFrame

    Args:
        file_pointer (FilePointer): File Pointer to a parquet file
        **kwargs: Additional arguments to pass to pandas read_parquet method

    Returns:
        Pandas DataFrame with the data from the parquet file
    """
    return pd.read_parquet(file_pointer, **kwargs)
