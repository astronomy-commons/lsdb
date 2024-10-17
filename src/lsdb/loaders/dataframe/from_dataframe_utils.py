from datetime import datetime, timezone
from typing import List, Tuple

import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask import delayed
from hats.io import paths
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb
from lsdb.dask.divisions import get_pixels_divisions


def _generate_dask_dataframe(
    pixel_dfs: List[npd.NestedFrame], pixels: List[HealpixPixel], use_pyarrow_types: bool = True
) -> Tuple[nd.NestedFrame, int]:
    """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

    Args:
        pixel_dfs (List[npd.NestedFrame]): The list of HEALPix pixel Dataframes
        pixels (List[HealpixPixel]): The list of HEALPix pixels in the catalog
        use_pyarrow_types (bool): If True, use pyarrow types. Defaults to True.

    Returns:
        The catalog's Dask Dataframe and its total number of rows.
    """
    pixel_dfs = [_convert_dtypes_to_pyarrow(df) for df in pixel_dfs] if use_pyarrow_types else pixel_dfs
    schema = pixel_dfs[0].iloc[:0, :].copy() if len(pixels) > 0 else []
    delayed_dfs = [delayed(df) for df in pixel_dfs]
    divisions = get_pixels_divisions(pixels)
    ddf = nd.NestedFrame.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
    return ddf, len(ddf)


def _convert_dtypes_to_pyarrow(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the columns (and index) of a Pandas DataFrame to pyarrow types.

    Args:
        df (pd.DataFrame): A Pandas DataFrame

    Returns:
        A new DataFrame, with columns of pyarrow types. The return value is a
        shallow copy of the initial DataFrame to avoid copying the data.
    """
    new_series = {}
    df_index = df.index.astype(pd.ArrowDtype(pa.int64()))
    for column in df.columns:
        pa_array = pa.array(df[column], from_pandas=True)
        series = pd.Series(pa_array, dtype=pd.ArrowDtype(pa_array.type), copy=False, index=df_index)
        new_series[column] = series
    return pd.DataFrame(new_series, index=df_index, copy=False)


def _append_partition_information_to_dataframe(
    dataframe: npd.NestedFrame, pixel: HealpixPixel
) -> npd.NestedFrame:
    """Append partitioning information to a HEALPix dataframe

    Args:
        dataframe (pd.Dataframe): A HEALPix's pandas dataframe
        pixel (HealpixPixel): The HEALPix pixel for the current partition

    Returns:
        The dataframe for a HEALPix, with data points and respective partition information.
    """
    columns_to_assign = {
        paths.PARTITION_ORDER: pixel.order,
        paths.PARTITION_DIR: pixel.dir,
        paths.PARTITION_PIXEL: pixel.pixel,
    }
    column_types = {
        paths.PARTITION_ORDER: np.uint8,
        paths.PARTITION_DIR: np.uint64,
        paths.PARTITION_PIXEL: np.uint64,
    }
    dataframe = dataframe.assign(**columns_to_assign).astype(column_types)
    return _order_partition_dataframe_columns(dataframe)


def _format_margin_partition_dataframe(dataframe: npd.NestedFrame) -> npd.NestedFrame:
    """Finalizes the dataframe for a margin catalog partition

    Args:
        dataframe (pd.DataFrame): The partition dataframe

    Returns:
        The dataframe for a margin partition with the data points and
        the respective pixel information.
    """
    dataframe = dataframe.drop(columns=["margin_pixel"])
    rename_columns = {
        paths.PARTITION_ORDER: f"margin_{paths.PARTITION_ORDER}",
        paths.PARTITION_DIR: f"margin_{paths.PARTITION_DIR}",
        paths.PARTITION_PIXEL: f"margin_{paths.PARTITION_PIXEL}",
        "partition_order": paths.PARTITION_ORDER,
        "partition_pixel": paths.PARTITION_PIXEL,
    }
    dataframe.rename(columns=rename_columns, inplace=True)
    dir_column = np.floor_divide(dataframe[paths.PARTITION_PIXEL].to_numpy(), 10000) * 10000
    dataframe[paths.PARTITION_DIR] = dir_column
    dataframe = dataframe.astype(
        {
            paths.PARTITION_ORDER: np.uint8,
            paths.PARTITION_DIR: np.uint64,
            paths.PARTITION_PIXEL: np.uint64,
        }
    )
    dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN).sort_index()
    return _order_partition_dataframe_columns(dataframe)


def _order_partition_dataframe_columns(dataframe: npd.NestedFrame) -> npd.NestedFrame:
    """Reorder columns of a partition dataframe so that pixel information
    always appears in the same sequence

    Args:
        dataframe (pd.DataFrame): The partition dataframe

    Returns:
        The partition dataframe with the columns in the correct order.
    """
    order_of_columns = [
        f"margin_{paths.PARTITION_ORDER}",
        f"margin_{paths.PARTITION_DIR}",
        f"margin_{paths.PARTITION_PIXEL}",
        paths.PARTITION_ORDER,
        paths.PARTITION_DIR,
        paths.PARTITION_PIXEL,
    ]
    unordered_columns = [col for col in dataframe.columns if col not in order_of_columns]
    ordered_columns = [col for col in order_of_columns if col in dataframe.columns]
    return dataframe[unordered_columns + ordered_columns]


def _extra_property_dict(est_size_bytes: int):
    """Create a dictionary of additional fields to store in the properties file."""
    properties = {}
    properties["hats_builder"] = f"lsdb v{lsdb.__version__}"

    now = datetime.now(tz=timezone.utc)
    properties["hats_creation_date"] = now.strftime("%Y-%m-%dT%H:%M%Z")
    properties["hats_estsize"] = f"{int(est_size_bytes / 1024)}"
    properties["hats_release_date"] = "2024-09-18"
    properties["hats_version"] = "v0.1"

    return properties
