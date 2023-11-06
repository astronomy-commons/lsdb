import dataclasses
import os
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, Union

import pandas as pd
from hipscat.io import paths
from hipscat.io.file_io import file_io
from hipscat.io.write_metadata import write_json_file


def write_dataframe_to_parquet(df: pd.DataFrame, path: str):
    """Writes a pandas dataframe to parquet

    Args:
        df (pd.Dataframe): The pandas dataframe to write to disk
        path (str): The path to write the parquet file to
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    df.to_parquet(path)


def write_provenance_info(
    catalog_base_dir: file_io.FilePointer,
    dataset_info,
    tool_args: dict,
    storage_options: Union[Dict[Any, Any], None] = None,
):
    """Write a provenance_info.json file with the available catalog creation metadata

    Args:
        catalog_base_dir (str): base directory for catalog, where file will be written
        dataset_info (:obj:`BaseCatalogInfo`) base metadata for the catalog
        tool_args (:obj:`dict`): dictionary of additional arguments provided by the tool creating
            this catalog.
        storage_options: dictionary that contains abstract filesystem credentials
    """
    metadata = dataclasses.asdict(dataset_info)
    metadata["version"] = version("lsdb")
    now = datetime.now()
    metadata["generation_date"] = now.strftime("%Y.%m.%d")
    metadata["tool_args"] = tool_args
    metadata_pointer = paths.get_provenance_pointer(catalog_base_dir)
    write_json_file(metadata, metadata_pointer, storage_options=storage_options)
