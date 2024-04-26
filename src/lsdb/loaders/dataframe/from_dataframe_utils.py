from typing import List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
from hipscat.catalog import PartitionInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN

from lsdb.dask.divisions import get_pixels_divisions


def _generate_dask_dataframe(
    pixel_dfs: List[pd.DataFrame], pixels: List[HealpixPixel]
) -> Tuple[dd.core.DataFrame, int]:
    """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

    Args:
        pixel_dfs (List[pd.DataFrame]): The list of HEALPix pixel Dataframes
        pixels (List[HealpixPixel]): The list of HEALPix pixels in the catalog

    Returns:
        The catalog's Dask Dataframe and its total number of rows.
    """
    schema = pixel_dfs[0].iloc[:0, :].copy() if len(pixels) > 0 else []
    divisions = get_pixels_divisions(pixels)
    delayed_dfs = [delayed(df) for df in pixel_dfs]
    ddf = dd.io.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
    return ddf if isinstance(ddf, dd.core.DataFrame) else ddf.to_frame(), len(ddf)


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
