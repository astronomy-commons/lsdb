import pandas as pd
import pyarrow.parquet as pq
from hipscat.io import FilePointer, file_io
from pyarrow import Schema


def read_parquet_schema(file_pointer: FilePointer, storage_options: dict = None) -> Schema:
    """Reads the schema from a parquet file

    Args:
        file_pointer (FilePointer): File Pointer to a parquet file

    Returns:
        PyArrow schema object with the schema of the parquet file
    """
    fs, _ = file_io.file_pointer.get_fs(file_pointer, storage_options=storage_options)
    return pq.read_schema(file_pointer, filesystem=fs)


def read_parquet_file_to_pandas(
        file_pointer: FilePointer, storage_options: dict = None, **kwargs
    ) -> pd.DataFrame:
    """Reads a parquet file to a pandas DataFrame

    Args:
        file_pointer (FilePointer): File Pointer to a parquet file
        **kwargs: Additional arguments to pass to pandas read_parquet method

    Returns:
        Pandas DataFrame with the data from the parquet file
    """
    return pd.read_parquet(file_pointer, storage_options=storage_options, **kwargs)
