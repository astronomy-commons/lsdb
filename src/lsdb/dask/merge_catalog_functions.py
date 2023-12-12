from __future__ import annotations

from typing import Sequence, Dict, List, TYPE_CHECKING

import pandas as pd
from dask.delayed import Delayed
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import healpix_to_hipscat_id, HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignment

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


def filter_by_hipscat_index_to_pixel(dataframe: pd.DataFrame, order: int, pixel: int) -> pd.DataFrame:
    """Filters a catalog dataframe to the points within a specified HEALPix pixel using the hipscat index

    Args:
        dataframe (pd.DataFrame): The dataframe to filter
        order (int): The order of the HEALPix pixel to filter to
        pixel (int): The pixel number in NESTED numbering of the HEALPix pixel to filter to

    Returns:
        The filtered dataframe with only the rows that are within the specified HEALPix pixel
    """
    lower_bound = healpix_to_hipscat_id(order, pixel)
    upper_bound = healpix_to_hipscat_id(order, pixel + 1)
    filtered_df = dataframe[(dataframe.index >= lower_bound) & (dataframe.index < upper_bound)]
    return filtered_df


def get_healpix_pixels_from_alignment(join_pixels):
    left_pixels = [
        HealpixPixel(
            row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME], row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME]
        )
        for _, row in join_pixels.iterrows()
    ]
    right_pixels = [
        HealpixPixel(row[PixelAlignment.JOIN_ORDER_COLUMN_NAME], row[PixelAlignment.JOIN_PIXEL_COLUMN_NAME])
        for _, row in join_pixels.iterrows()
    ]
    return left_pixels, right_pixels


def generate_meta_df_for_joined_tables(
    catalogs: Sequence[Catalog],
    suffixes: Sequence[str],
    extra_columns: Dict[str, pd.Series] | None = None,
    index_name: str = HIPSCAT_ID_COLUMN,
) -> pd.DataFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs

    Creates an empty dataframe with the columns of each catalog appended with a suffix. Allows specifying
    extra columns that should also be added, and the name of the index of the resulting dataframe.

    Args:
        catalogs (Sequence[Catalog]): The catalogs to merge together
        suffixes (Sequence[Str]): The column suffixes to apply each catalog
        extra_columns (Dict[str, pd.Series]): Any additional columns to the merged catalogs
        index_name: The name of the index in the resulting DataFrame

    Returns:
    An empty dataframe with the columns of each catalog with their respective suffix, and any extra columns
    specified, with the index name set.
    """
    meta = {}
    for table, suffix in zip(catalogs, suffixes):
        for name, col_type in table.dtypes.items():
            meta[name + suffix] = pd.Series(dtype=col_type)
    if extra_columns is not None:
        meta.update(extra_columns)
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = index_name
    return meta_df


def get_partition_map_from_alignment_pixels(join_pixels: pd.DataFrame) -> DaskDFPixelMap:
    """Gets a dictionary mapping HEALPix pixel to index of pixel in the pixel_mapping of a `PixelAlignment`

    Args:
        join_pixels (pd.DataFrame): The pixel_mapping from a `PixelAlignment` object

    Returns:
        A dictionary mapping HEALPix pixel to the index that the pixel occurs in the pixel_mapping table
    """
    partition_map = {}
    for i, (_, row) in enumerate(join_pixels.iterrows()):
        pixel = HealpixPixel(
            order=row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME],
            pixel=row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME],
        )
        partition_map[pixel] = i
    return partition_map


def align_catalog_to_partitions(
    catalog: HealpixDataset,
    pixels: pd.DataFrame,
    order_col: str = "Norder",
    pixel_col: str = "Npix",
) -> List[Delayed]:
    """Aligns the partitions of a Catalog to a dataframe with HEALPix pixels in each row

    Args:
        catalog: the catalog to align
        pixels: the dataframe specifying the order of partitions
        order_col: the column name of the HEALPix order in the dataframe
        pixel_col: the column name of the HEALPix pixel in the dataframe

    Returns:
        A list of dask delayed objects, each one representing the data in a HEALPix pixel in the
        order they appear in the input dataframe

    """
    dfs = catalog.to_delayed()
    partitions = pixels.apply(
        lambda row: dfs[catalog.get_partition_index(row[order_col], row[pixel_col])],
        axis=1,
    )
    partitions_list = partitions.to_list()
    return partitions_list


def align_catalogs_to_alignment_mapping(join_pixels, left, right):
    left_aligned_to_join_partitions = align_catalog_to_partitions(
        left,
        join_pixels,
        order_col=PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    )
    right_aligned_to_join_partitions = align_catalog_to_partitions(
        right,
        join_pixels,
        order_col=PixelAlignment.JOIN_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
    )
    return left_aligned_to_join_partitions, right_aligned_to_join_partitions
