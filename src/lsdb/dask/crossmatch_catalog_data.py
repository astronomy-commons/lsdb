from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple, Type, cast

import dask
import dask.dataframe as dd
import pandas as pd
from dask.delayed import Delayed
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN, healpix_to_hipscat_id
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog

builtin_crossmatch_algorithms = {BuiltInCrossmatchAlgorithm.KD_TREE: KdTreeCrossmatch}


# pylint: disable=too-many-arguments
@dask.delayed
def perform_crossmatch(
    algorithm,
    left_df,
    right_df,
    left_order,
    left_pixel,
    right_order,
    right_pixel,
    left_hc_structure,
    right_hc_structure,
    suffixes,
    **kwargs,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if right_order > left_order:
        left_df = filter_by_hipscat_index_to_pixel(left_df, right_order, right_pixel)
    return algorithm(
        left_df,
        right_df,
        left_order,
        left_pixel,
        right_order,
        right_pixel,
        left_hc_structure,
        right_hc_structure,
        suffixes,
    ).crossmatch(**kwargs)


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


# pylint: disable=too-many-locals
def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    suffixes: Tuple[str, str],
    algorithm: Type[AbstractCrossmatchAlgorithm]
    | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE,
    **kwargs,
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Args:
        left (Catalog): the left catalog to perform the cross-match on
        right (Catalog): the right catalog to perform the cross-match on
        suffixes (Tuple[str,str]): the suffixes to append to the column names from the left and
            right catalogs respectively
        algorithm (BuiltInCrossmatchAlgorithm | Callable): The algorithm to use to perform the
            crossmatch. Can be specified using a string for a built-in algorithm, or a custom
            method. For more details, see `crossmatch` method in the `Catalog` class.
        **kwargs: Additional arguments to pass to the cross-match algorithm

    Returns:
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    crossmatch_algorithm = get_crossmatch_algorithm(algorithm)

    # perform alignment on the two catalogs
    alignment = align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER,
    )
    join_pixels = alignment.pixel_mapping

    # align partitions from the catalogs to match the pixel alignment
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

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_orders = [row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    left_pixels = [row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    right_orders = [row[PixelAlignment.JOIN_ORDER_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    right_pixels = [row[PixelAlignment.JOIN_PIXEL_COLUMN_NAME] for _, row in join_pixels.iterrows()]

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = [
        perform_crossmatch(
            crossmatch_algorithm,
            left_df,
            right_df,
            left_order,
            left_pixel,
            right_order,
            right_pixel,
            left.hc_structure,
            right.hc_structure,
            suffixes,
            **kwargs,
        )
        for left_df, right_df, left_order, left_pixel, right_order, right_pixel in zip(
            left_aligned_to_join_partitions,
            right_aligned_to_join_partitions,
            left_orders,
            left_pixels,
            right_orders,
            right_pixels,
        )
    ]

    # generate dask df partition map from alignment
    partition_map = get_partition_map_from_alignment_pixels(join_pixels)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_joined_tables(
        [left, right], suffixes, extra_columns=crossmatch_algorithm.extra_columns
    )

    # create dask df from delayed partitions
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    ddf = cast(dd.DataFrame, ddf)

    return ddf, partition_map, alignment


def generate_meta_df_for_joined_tables(
    catalogs: Sequence[Catalog],
    suffixes: Sequence[str],
    extra_columns: pd.DataFrame | None = None,
    index_name: str = HIPSCAT_ID_COLUMN,
) -> pd.DataFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs

    Creates an empty dataframe with the columns of each catalog appended with a suffix. Allows specifying
    extra columns that should also be added, and the name of the index of the resulting dataframe.

    Args:
        catalogs (Sequence[Catalog]): The catalogs to merge together
        suffixes (Sequence[Str]): The column suffixes to apply each catalog
        extra_columns (pd.Dataframe): Any additional columns to the merged catalogs
        index_name (str): The name of the index in the resulting DataFrame

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


def get_crossmatch_algorithm(
    algorithm: Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm,
) -> Type[AbstractCrossmatchAlgorithm]:
    """Gets the function to perform a cross-match algorithm

    Args:
        algorithm: The algorithm to use to perform the cross-match. Can be specified using a string
        for a built-in algorithm, or a custom method.

    Returns:
        The function to perform the specified crossmatch. Either by looking up the method for a
        built-in algorithm, or returning the custom function.
    """
    if isinstance(algorithm, BuiltInCrossmatchAlgorithm):
        return builtin_crossmatch_algorithms[algorithm]
    if issubclass(algorithm, AbstractCrossmatchAlgorithm):
        return algorithm
    raise TypeError("algorithm must be either callable or a string for a builtin algorithm")


def align_catalog_to_partitions(
    catalog: Catalog,
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
