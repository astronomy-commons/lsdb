import os
from typing import Any, Dict, Union

import hipscat as hc
import pandas as pd


def write_dataframe_to_parquet(
        dataframe: pd.DataFrame, path: str, storage_options: Union[Dict[Any, Any], None] = None, **kwargs
):
    """Writes a pandas dataframe to parquet, creating the respective parent directory
     if it does not yet exist

    Args:
        dataframe (pd.Dataframe): The pandas dataframe to write to disk
        path (str): The path to write the parquet file to
        storage_options (dict): dictionary that contains abstract filesystem credentials
        **kwargs: Arguments to pass to the parquet write operation
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    base_catalog_dir_fp = hc.io.get_file_pointer_from_path(parent_dir)
    hc.io.file_io.make_directory(base_catalog_dir_fp, exist_ok=True, storage_options=storage_options)
    hc.io.file_io.write_dataframe_to_parquet(dataframe, path, storage_options, **kwargs)
