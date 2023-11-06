import os

import pandas as pd


def write_dataframe_to_parquet(df: pd.DataFrame, path: str):
    """Writes a pandas dataframe to parquet

    Args:
        df (pd.Dataframe): The pandas dataframe to write to disk
        path (str): The path to write the parquet file to
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    df.to_parquet(path)
