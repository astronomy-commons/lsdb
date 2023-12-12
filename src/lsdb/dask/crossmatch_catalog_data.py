from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.dask.merge_catalog_functions import (
    filter_by_hipscat_index_to_pixel,
    get_partition_map_from_alignment_pixels,
    generate_meta_df_for_joined_tables,
    align_catalogs_to_alignment_mapping,
)
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
    left_aligned_partitions, right_aligned_partitions = align_catalogs_to_alignment_mapping(
        join_pixels, left, right
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
            left_aligned_partitions,
            right_aligned_partitions,
            left_orders,
            left_pixels,
            right_orders,
            right_pixels,
        )
    ]

    # generate dask df partition map from alignment
    partition_map = get_partition_map_from_alignment_pixels(join_pixels)

    # generate meta table structure for dask df
    extra_columns = {crossmatch_algorithm.DISTANCE_COLUMN_NAME: pd.Series(dtype=np.dtype("float64"))}
    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes, extra_columns=extra_columns)

    # create dask df from delayed partitions
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    ddf = cast(dd.DataFrame, ddf)

    return ddf, partition_map, alignment


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
