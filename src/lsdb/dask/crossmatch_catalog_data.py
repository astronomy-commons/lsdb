from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import healpix_to_hipscat_id
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.crossmatch_algorithms import \
    BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import kd_tree_crossmatch

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog, DaskDFPixelMap

builtin_crossmatch_algorithms = {BuiltInCrossmatchAlgorithm.KD_TREE: kd_tree_crossmatch}

CrossmatchAlgorithmType = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


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
        lower_bound = healpix_to_hipscat_id(right_order, right_pixel)
        upper_bound = healpix_to_hipscat_id(right_order, right_pixel + 1)
        left_df = left_df[(left_df.index > lower_bound) & (left_df.index < upper_bound)]
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
        **kwargs,
    )


def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    suffixes: Tuple[str, str] | None = None,
    algorithm: CrossmatchAlgorithmType
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
    left_orders = [
        row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME]
        for _, row in join_pixels.iterrows()
    ]
    left_pixels = [
        row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME]
        for _, row in join_pixels.iterrows()
    ]
    right_orders = [
        row[PixelAlignment.JOIN_ORDER_COLUMN_NAME] for _, row in join_pixels.iterrows()
    ]
    right_pixels = [
        row[PixelAlignment.JOIN_PIXEL_COLUMN_NAME] for _, row in join_pixels.iterrows()
    ]

    # perform the crossmatch on each partition pairing using dask delayed for laziness
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
    partition_map = {}
    for i, (_, row) in enumerate(join_pixels.iterrows()):
        pixel = HealpixPixel(
            order=row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME],
            pixel=row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME],
        )
        partition_map[pixel] = i

    # generate meta table structure for dask df
    meta = {}
    for name, col_type in left.dtypes.items():
        meta[name + suffixes[0]] = pd.Series(dtype=col_type)
    for name, col_type in right.dtypes.items():
        meta[name + suffixes[1]] = pd.Series(dtype=col_type)
    meta["_DIST"] = pd.Series(dtype=np.dtype("float64"))
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = "_hipscat_index"

    # create dask df from delayed partitions
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)

    return ddf, partition_map, alignment


def get_crossmatch_algorithm(
    algorithm: CrossmatchAlgorithmType | BuiltInCrossmatchAlgorithm,
) -> CrossmatchAlgorithmType:
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
    if callable(algorithm):
        return algorithm
    raise TypeError(
        "algorithm must be either callable or a string for a builtin algorithm"
    )


def align_catalog_to_partitions(
    catalog: Catalog,
    pixels: pd.DataFrame,
    order_col: str = "Norder",
    pixel_col: str = "Npix",
) -> List[dask.delayed.Delayed]:
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
