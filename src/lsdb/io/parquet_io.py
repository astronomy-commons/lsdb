import pandas as pd
from hipscat.io import FilePointer
from pyarrow import Schema
import pyarrow.parquet as pq


def read_parquet_schema(file_pointer: FilePointer) -> Schema:
    return pq.read_schema(file_pointer)


def read_parquet_file_to_pandas(file_pointer: FilePointer, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(file_pointer, **kwargs)
