from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type

import dask
import dask.dataframe as dd
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    construct_catalog_args,
    filter_by_hipscat_index_to_pixel,
    generate_meta_df_for_joined_tables,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog

builtin_crossmatch_algorithms = {BuiltInCrossmatchAlgorithm.KD_TREE: KdTreeCrossmatch}


# pylint: disable=too-many-arguments
@dask.delayed
def perform_crossmatch(
    left_df,
    right_df,
    left_pix,
    right_pix,
    left_hc_structure,
    right_hc_structure,
    algorithm,
    suffixes,
    **kwargs,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if right_pix.order > left_pix.order:
        left_df = filter_by_hipscat_index_to_pixel(left_df, right_pix.order, right_pix.pixel)
    return algorithm(
        left_df,
        right_df,
        left_pix.order,
        left_pix.pixel,
        right_pix.order,
        right_pix.pixel,
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

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels)],
        perform_crossmatch,
        crossmatch_algorithm,
        suffixes,
        **kwargs,
    )

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_joined_tables(
        [left, right], suffixes, extra_columns=crossmatch_algorithm.extra_columns
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


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
