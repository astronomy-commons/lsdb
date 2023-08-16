from __future__ import annotations

from typing import Callable, Tuple, TYPE_CHECKING

import dask
import numpy as np
import pandas as pd
import dask.dataframe as dd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import kd_tree_crossmatch

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog, DaskDFPixelMap

crossmatch_algorithms = {
    BuiltInCrossmatchAlgorithm.KD_TREE: kd_tree_crossmatch
}

CrossmatchAlgorithmType = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def crossmatch_catalog_data(
        left: Catalog,
        right: Catalog,
        suffixes: Tuple[str, str] | None = None,
        algorithm: CrossmatchAlgorithmType | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    crossmatch_algorithm = dask.delayed(get_crossmatch_algorithm(algorithm))
    alignment = align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER
    )
    join_pixels = alignment.pixel_mapping
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
    orders = [row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    pixels = [row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    joined_partitions = [
        crossmatch_algorithm(
            left_df, right_df, order, pixel, left.hc_structure, right.hc_structure, suffixes
        )
        for left_df, right_df, order, pixel in
        zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, orders, pixels)
    ]
    partition_map = {}
    for i, (_, row) in enumerate(join_pixels.iterrows()):
        pixel = HealpixPixel(order=row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME], pixel=row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME])
        partition_map[pixel] = i
    meta = {}
    for name, t in left._ddf.dtypes.items():
        meta[name + suffixes[0]] = pd.Series(dtype=t)
    for name, t in right._ddf.dtypes.items():
        meta[name + suffixes[1]] = pd.Series(dtype=t)
    meta['_DIST'] = pd.Series(dtype=np.dtype("float64"))
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = "_hipscat_index"
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    return ddf, partition_map, alignment


def get_crossmatch_algorithm(algorithm: CrossmatchAlgorithmType | BuiltInCrossmatchAlgorithm) -> CrossmatchAlgorithmType:
    if isinstance(algorithm, BuiltInCrossmatchAlgorithm):
        return crossmatch_algorithms[algorithm]
    elif callable(algorithm):
        return algorithm
    raise TypeError("algorithm must be either callable or a string for a builtin algorithm")


def align_catalog_to_partitions(
        catalog: Catalog,
        pixels: pd.DataFrame,
        order_col: str = "Norder",
        pixel_col: str = "Npix"
) -> dd.core.DataFrame:
    dfs = catalog._ddf.to_delayed()
    partitions = pixels.apply(lambda row: dfs[
        catalog.get_partition_index(row[order_col], row[pixel_col])], axis=1)
    partitions_list = partitions.to_list()
    return partitions_list
