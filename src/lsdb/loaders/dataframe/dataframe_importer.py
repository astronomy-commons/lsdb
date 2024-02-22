from typing import List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
from hipscat.catalog import PartitionInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN

from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import DaskDFPixelMap


class DataframeImporter:
    @classmethod
    def _generate_dask_dataframe(
        cls, pixel_dfs: List[pd.DataFrame], ddf_pixel_map: DaskDFPixelMap,
    ) -> Tuple[dd.DataFrame, int]:
        """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

        Args:
            pixel_dfs (List[pd.DataFrame]): The list of HEALPix pixel Dataframes
            ddf_pixel_map (DaskDD

        Returns:
            The catalog's Dask Dataframe and its total number of rows.
        """
        schema = pixel_dfs[0].iloc[:0, :].copy()
        divisions = get_pixels_divisions(list(ddf_pixel_map.keys()))
        delayed_dfs = [delayed(df) for df in pixel_dfs]
        ddf = dd.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
        return ddf if isinstance(ddf, dd.DataFrame) else ddf.to_frame(), len(ddf)

    @classmethod
    def _append_partition_information_to_dataframe(
        cls, dataframe: pd.DataFrame, pixel: HealpixPixel
    ) -> pd.DataFrame:
        """Appends partitioning information to a HEALPix dataframe

        Args:
            dataframe (pd.Dataframe): A HEALPix's pandas dataframe
            pixel (HealpixPixel): The HEALPix pixel for the current partition

        Returns:
            The dataframe for a HEALPix, with data points and respective partition information.
        """
        ordered_columns = ["Norder", "Dir", "Npix"]
        # Generate partition information
        dataframe["Norder"] = pixel.order
        dataframe["Npix"] = pixel.pixel
        dataframe["Dir"] = pixel.dir
        # Force new column types to int
        dataframe[ordered_columns] = dataframe[ordered_columns].astype(int)
        # Reorder the columns to match full path
        return dataframe[[col for col in dataframe.columns if col not in ordered_columns] + ordered_columns]

    @classmethod
    def _finalize_margin_partition_dataframe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.drop(columns=["margin_pixel"])
        rename_columns = {
            PartitionInfo.METADATA_ORDER_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_ORDER_COLUMN_NAME}",
            PartitionInfo.METADATA_DIR_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_DIR_COLUMN_NAME}",
            PartitionInfo.METADATA_PIXEL_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_PIXEL_COLUMN_NAME}",
            "partition_order": PartitionInfo.METADATA_ORDER_COLUMN_NAME,
            "partition_pixel": PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
        }
        dataframe.rename(columns=rename_columns, inplace=True)
        dir_column = (
            np.floor_divide(dataframe[PartitionInfo.METADATA_PIXEL_COLUMN_NAME].values, 10000) * 10000
        )
        dataframe[PartitionInfo.METADATA_DIR_COLUMN_NAME] = dir_column
        dataframe = dataframe.astype(
            {
                PartitionInfo.METADATA_ORDER_COLUMN_NAME: np.uint8,
                PartitionInfo.METADATA_DIR_COLUMN_NAME: np.uint64,
                PartitionInfo.METADATA_PIXEL_COLUMN_NAME: np.uint64,
            }
        )
        dataframe = dataframe.set_index(HIPSCAT_ID_COLUMN).sort_index()

        # TODO: Improve this ordering of columns part
        ordered_columns = ["margin_Norder", "margin_Dir", "margin_Npix", "Norder", "Npix", "Dir"]
        return dataframe[[col for col in dataframe.columns if col not in ordered_columns] + ordered_columns]
