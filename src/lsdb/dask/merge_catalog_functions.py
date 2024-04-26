from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, cast

import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask.delayed import Delayed
from hipscat.catalog import PartitionInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN, healpix_to_hipscat_id
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def concat_partition_and_margin(
    partition: pd.DataFrame, margin: pd.DataFrame | None, right_columns: List[str]
) -> pd.DataFrame:
    """Concatenates a partition and margin dataframe together

    Args:
        partition (pd.DataFrame): The partition dataframe
        margin (pd.DataFrame): The margin dataframe

    Returns:
        The concatenated dataframe with the partition on top and the margin on the bottom
    """
    if margin is None:
        return partition

    hive_columns = [
        PartitionInfo.METADATA_ORDER_COLUMN_NAME,
        PartitionInfo.METADATA_DIR_COLUMN_NAME,
        PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
    ]
    # Remove the Norder/Dir/Npix columns (used only for partitioning the margin itself),
    # and rename the margin_Norder/Dir/Npix to take their place.
    margin_columns_no_hive = [col for col in margin.columns if col not in hive_columns]
    rename_columns = {
        f"margin_{PartitionInfo.METADATA_ORDER_COLUMN_NAME}": PartitionInfo.METADATA_ORDER_COLUMN_NAME,
        f"margin_{PartitionInfo.METADATA_DIR_COLUMN_NAME}": PartitionInfo.METADATA_DIR_COLUMN_NAME,
        f"margin_{PartitionInfo.METADATA_PIXEL_COLUMN_NAME}": PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
    }
    margin_renamed = margin[margin_columns_no_hive].rename(columns=rename_columns)
    margin_filtered = margin_renamed[right_columns]
    joined_df = pd.concat([partition, margin_filtered]) if margin_filtered is not None else partition
    return joined_df


def align_catalogs(left: Catalog, right: Catalog) -> PixelAlignment:
    """Aligns two catalogs, also using the right catalog's margin if it exists

    Args:
        left (lsdb.Catalog): The left catalog to align
        right (lsdb.Catalog): The right catalog to align
    Returns:
        The PixelAlignment object from aligning the catalogs
    """

    if right.margin is not None:
        right_tree = align_trees(
            right.hc_structure.pixel_tree,
            right.margin.hc_structure.pixel_tree,
            alignment_type=PixelAlignmentType.OUTER,
        ).pixel_tree
    else:
        right_tree = right.hc_structure.pixel_tree
    return align_trees(left.hc_structure.pixel_tree, right_tree, alignment_type=PixelAlignmentType.INNER)


def align_and_apply(
    catalog_mappings: List[Tuple[HealpixDataset | None, List[HealpixPixel]]], func: Callable, *args, **kwargs
) -> List[Delayed]:
    """Aligns catalogs to a given ordering of pixels and applies a function each set of aligned partitions

    Args:
        catalog_mappings (List[Tuple[HealpixDataset, List[HealpixPixel]]]): The catalogs and their
            corresponding ordering of pixels to align the partitions to. Catalog cane be None, in which case
            None will be passed to the function for each partition. Each list of pixels should be the same
            length. Example input:
            [(catalog, pixels), (catalog2, pixels2), ...]
        func (Callable): The function to apply to the aligned catalogs. The function should take the
            aligned partitions of the catalogs as dataframes as the first arguments, followed by the healpix
            pixel of each partition, the hc_structures of the catalogs, and any additional arguments and
            keyword arguments. For example::

                def func(
                    cat1_partition_df,
                    cat2_partition_df,
                    cat1_pixel,
                    cat2_pixel,
                    cat1_hc_structure,
                    cat2_hc_structure,
                    *args,
                    **kwargs
                ):
                    ...

        *args: Additional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function

    Returns:
        A list of delayed objects, each one representing the result of the function applied to the
        aligned partitions of the catalogs
    """

    # aligns the catalog's partitions to the given pixels for each catalog
    aligned_partitions = [align_catalog_to_partitions(cat, pixels) for (cat, pixels) in catalog_mappings]

    # gets the pixels and hc_structures to pass to the function
    pixels = [pixels for (_, pixels) in catalog_mappings]
    hc_structures = [cat.hc_structure if cat is not None else None for (cat, _) in catalog_mappings]

    # defines an inner function that can be vectorized to apply the given function to each of the partitions
    # with the additional arguments including as the hc_structures and any specified additional arguments
    def apply_func(*partitions_and_pixels):
        return func(*partitions_and_pixels, *hc_structures, *args, **kwargs)

    resulting_partitions = np.vectorize(apply_func)(*aligned_partitions, *pixels)
    return resulting_partitions


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


def construct_catalog_args(
    partitions: List[Delayed], meta_df: pd.DataFrame, alignment: PixelAlignment
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Constructs the arguments needed to create a catalog from a list of delayed partitions

    Args:
        partitions (List[Delayed]): The list of delayed partitions to create the catalog from
        meta_df (pd.DataFrame): The dask meta schema for the partitions
        alignment (PixelAlignment): The alignment used to create the delayed partitions

    Returns:
        A tuple of (ddf, partition_map, alignment) with the dask dataframe, the partition map, and the
        alignment needed to create the catalog
    """
    # generate dask df partition map from alignment
    partition_map = get_partition_map_from_alignment_pixels(alignment.pixel_mapping)

    # create dask df from delayed partitions
    divisions = get_pixels_divisions(list(partition_map.keys()))
    ddf = dd.io.from_delayed(partitions, meta=meta_df, divisions=divisions)
    ddf = cast(dd.core.DataFrame, ddf)
    return ddf, partition_map, alignment


def get_healpix_pixels_from_alignment(
    alignment: PixelAlignment,
) -> Tuple[List[HealpixPixel], List[HealpixPixel]]:
    """Gets the list of primary and join pixels as the HealpixPixel class from a PixelAlignment

    Args:
        alignment (PixelAlignment): the PixelAlignment to get pixels from

    Returns:
        a tuple of (primary_pixels, join_pixels) with lists of HealpixPixel objects
    """
    pixel_mapping = alignment.pixel_mapping
    make_pixel = np.vectorize(HealpixPixel)
    left_pixels = make_pixel(
        pixel_mapping[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME],
        pixel_mapping[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME],
    )
    right_pixels = make_pixel(
        pixel_mapping[PixelAlignment.JOIN_ORDER_COLUMN_NAME],
        pixel_mapping[PixelAlignment.JOIN_PIXEL_COLUMN_NAME],
    )
    return list(left_pixels), list(right_pixels)


def generate_meta_df_for_joined_tables(
    catalogs: Sequence[Catalog],
    suffixes: Sequence[str],
    extra_columns: pd.DataFrame | None = None,
    index_name: str = HIPSCAT_ID_COLUMN,
    index_type: npt.DTypeLike = np.uint64,
) -> pd.DataFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs

    Creates an empty dataframe with the columns of each catalog appended with a suffix. Allows specifying
    extra columns that should also be added, and the name of the index of the resulting dataframe.

    Args:
        catalogs (Sequence[lsdb.Catalog]): The catalogs to merge together
        suffixes (Sequence[Str]): The column suffixes to apply each catalog
        extra_columns (pd.Dataframe): Any additional columns to the merged catalogs
        index_name (str): The name of the index in the resulting DataFrame
        index_type (npt.DTypeLike): The type of the index in the resulting DataFrame

    Returns:
    An empty dataframe with the columns of each catalog with their respective suffix, and any extra columns
    specified, with the index name set.
    """
    meta = {}
    # Construct meta for crossmatched catalog columns
    for table, suffix in zip(catalogs, suffixes):
        for name, col_type in table.dtypes.items():
            meta[name + suffix] = pd.Series(dtype=col_type)
    # Construct meta for crossmatch result columns
    if extra_columns is not None:
        meta.update(extra_columns)
    index = pd.Index(pd.Series(dtype=index_type), name=index_name)
    meta_df = pd.DataFrame(meta, index)
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
    catalog: HealpixDataset | None, pixels: List[HealpixPixel]
) -> List[Delayed | None]:
    """Aligns the partitions of a Catalog to a dataframe with HEALPix pixels in each row

    Args:
        catalog: the catalog to align
        pixels: the list of HealpixPixels specifying the order of partitions

    Returns:
        A list of dask delayed objects, each one representing the data in a HEALPix pixel in the
        order they appear in the input dataframe

    """
    if catalog is None:
        return [None] * len(pixels)
    dfs = catalog.to_delayed()
    get_partition = np.vectorize(
        lambda pix: (
            dfs[catalog.get_partition_index(pix.order, pix.pixel)]
            if pix in catalog.hc_structure.pixel_tree
            else None
        )
    )
    partitions = get_partition(pixels)
    return list(partitions)
