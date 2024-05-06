from typing import List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask import delayed
from hipscat.catalog import PartitionInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN

from lsdb.dask.divisions import get_pixels_divisions


def _generate_dask_dataframe(
    pixel_dfs: List[pd.DataFrame], pixels: List[HealpixPixel], use_pyarrow_types: bool = True
) -> Tuple[dd.core.DataFrame, int]:
    """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

    Args:
        pixel_dfs (List[pd.DataFrame]): The list of HEALPix pixel Dataframes
        pixels (List[HealpixPixel]): The list of HEALPix pixels in the catalog
        use_pyarrow_types (bool): If True, use pyarrow types. Defaults to "True".

    Returns:
        The catalog's Dask Dataframe and its total number of rows.
    """
    schema = pixel_dfs[0].iloc[:0, :].copy() if len(pixels) > 0 else []
    delayed_dfs = [delayed(df) for df in pixel_dfs]
    divisions = get_pixels_divisions(pixels)
    ddf = dd.io.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
    ddf = ddf if isinstance(ddf, dd.core.DataFrame) else ddf.to_frame()
    ddf = _convert_ddf_types_to_pyarrow(ddf) if use_pyarrow_types else ddf
    return ddf, len(ddf)


# pylint: disable=protected-access
def _convert_ddf_types_to_pyarrow(ddf: dd.DataFrame) -> dd.DataFrame:
    """Convert a Dask DataFrame to pyarrow types.

    Args:
        ddf (dd.DataFrame): A Dask DataFrame

    Returns:
        A new dask DataFrame, where columns, index and schema have been
        converted to use pyarrow types.
    """
    # Convert schema types according to the backend
    pyarrow_meta = _convert_dtypes_to_pyarrow(ddf._meta)
    # Apply the new schema to the dask dataframe
    ddf = ddf.astype(pyarrow_meta.dtypes)
    # Update the hipscat index data type, which is not handled automatically
    ddf.index = ddf.index.astype(pd.ArrowDtype(pa.uint64()))
    # Finally, set the new schema as the dataframe meta
    ddf._meta = pyarrow_meta
    return ddf


def _convert_dtypes_to_pyarrow(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the columns of a Pandas DataFrame to pyarrow types. It
    does not update the type of the DataFrame index.

    Args:
        df (pd.DataFrame): A Pandas DataFrame

    Returns:
        A new DataFrame, with columns of pyarrow types. The return value is a
        shallow copy of the initial DataFrame to avoid copying the data.
    """
    new_series = {}
    for column in df.columns:
        try:
            pa_array = pa.array(df[column], from_pandas=True)
        except Exception as exc:
            raise ValueError(f"Could not convert column {column} to a pyarrow type") from exc
        # The LargeString type is not recommended. Prevent any strings from being cast to this type.
        pyarrow_dtype = pa.string() if isinstance(pa_array, pa.LargeStringArray) else pa_array.type
        series = pd.Series(pa_array, dtype=pd.ArrowDtype(pyarrow_dtype), copy=False, index=df.index)
        new_series[column] = series
    return pd.DataFrame(new_series, index=df.index, copy=False)


def _append_partition_information_to_dataframe(dataframe: pd.DataFrame, pixel: HealpixPixel) -> pd.DataFrame:
    """Append partitioning information to a HEALPix dataframe

    Args:
        dataframe (pd.Dataframe): A HEALPix's pandas dataframe
        pixel (HealpixPixel): The HEALPix pixel for the current partition

    Returns:
        The dataframe for a HEALPix, with data points and respective partition information.
    """
    columns_to_assign = {
        PartitionInfo.METADATA_ORDER_COLUMN_NAME: pixel.order,
        PartitionInfo.METADATA_DIR_COLUMN_NAME: pixel.dir,
        PartitionInfo.METADATA_PIXEL_COLUMN_NAME: pixel.pixel,
    }
    column_types = {
        PartitionInfo.METADATA_ORDER_COLUMN_NAME: np.uint8,
        PartitionInfo.METADATA_DIR_COLUMN_NAME: np.uint64,
        PartitionInfo.METADATA_PIXEL_COLUMN_NAME: np.uint64,
    }
    dataframe = dataframe.assign(**columns_to_assign).astype(column_types)
    return _order_partition_dataframe_columns(dataframe)


def _format_margin_partition_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Finalizes the dataframe for a margin catalog partition

    Args:
        dataframe (pd.DataFrame): The partition dataframe

    Returns:
        The dataframe for a margin partition with the data points and
        the respective pixel information.
    """
    dataframe = dataframe.drop(columns=["margin_pixel"])
    rename_columns = {
        PartitionInfo.METADATA_ORDER_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_ORDER_COLUMN_NAME}",
        PartitionInfo.METADATA_DIR_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_DIR_COLUMN_NAME}",
        PartitionInfo.METADATA_PIXEL_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_PIXEL_COLUMN_NAME}",
        "partition_order": PartitionInfo.METADATA_ORDER_COLUMN_NAME,
        "partition_pixel": PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
    }
    dataframe.rename(columns=rename_columns, inplace=True)
    dir_column = np.floor_divide(dataframe[PartitionInfo.METADATA_PIXEL_COLUMN_NAME].values, 10000) * 10000
    dataframe[PartitionInfo.METADATA_DIR_COLUMN_NAME] = dir_column
    dataframe = dataframe.astype(
        {
            PartitionInfo.METADATA_ORDER_COLUMN_NAME: np.uint8,
            PartitionInfo.METADATA_DIR_COLUMN_NAME: np.uint64,
            PartitionInfo.METADATA_PIXEL_COLUMN_NAME: np.uint64,
        }
    )
    dataframe = dataframe.set_index(HIPSCAT_ID_COLUMN).sort_index()
    return _order_partition_dataframe_columns(dataframe)


def _order_partition_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns of a partition dataframe so that pixel information
    always appears in the same sequence

    Args:
        dataframe (pd.DataFrame): The partition dataframe

    Returns:
        The partition dataframe with the columns in the correct order.
    """
    order_of_columns = [
        f"margin_{PartitionInfo.METADATA_ORDER_COLUMN_NAME}",
        f"margin_{PartitionInfo.METADATA_DIR_COLUMN_NAME}",
        f"margin_{PartitionInfo.METADATA_PIXEL_COLUMN_NAME}",
        PartitionInfo.METADATA_ORDER_COLUMN_NAME,
        PartitionInfo.METADATA_DIR_COLUMN_NAME,
        PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
    ]
    unordered_columns = [col for col in dataframe.columns if col not in order_of_columns]
    ordered_columns = [col for col in order_of_columns if col in dataframe.columns]
    return dataframe[unordered_columns + ordered_columns]
