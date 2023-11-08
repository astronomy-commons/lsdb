import os
from typing import Any, Dict, Union

import hipscat.io.file_io
import pandas as pd


def write_dataframe_to_parquet(
    dataframe: pd.DataFrame, path: str, storage_options: Union[Dict[Any, Any], None] = None
):
    """Writes a pandas dataframe to parquet, creating the respective
    parent directory if it does not yet exist

    Args:
        dataframe (pd.Dataframe): The pandas dataframe to write to disk
        path (str): The path to write the parquet file to
        storage_options (dict): dictionary that contains abstract filesystem credentials
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    hipscat.io.file_io.write_dataframe_to_parquet(dataframe, path, storage_options)
