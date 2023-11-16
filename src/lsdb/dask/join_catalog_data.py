from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import dask
import dask.dataframe as dd
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import healpix_to_hipscat_id, HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignmentType, PixelAlignment, align_trees

from lsdb.dask.crossmatch_catalog_data import align_catalog_to_partitions

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog, DaskDFPixelMap


@dask.delayed
def perform_join_on(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, left_pixel: HealpixPixel, right_pixel: HealpixPixel, suffixes: Tuple[str, str]):
    if right_pixel.order > left_pixel.order:
        lower_bound = healpix_to_hipscat_id(right_pixel.order, right_pixel.pixel)
        upper_bound = healpix_to_hipscat_id(right_pixel.order, right_pixel.pixel + 1)
        left = left[(left.index >= lower_bound) & (left.index < upper_bound)]
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)
    merged = left.reset_index().merge(right, left_on=left_on + suffixes[0], right_on=right_on + suffixes[1])
    merged.set_index("_hipscat_index", inplace=True)
    return merged


def join_catalog_data_on(
        left: Catalog,
        right: Catalog,
        left_on: str = None,
        right_on: str = None,
        suffixes: Tuple[str, str] | None = None
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
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
    joined_partitions = [perform_join_on(left_df, right_df, left_on, right_on, suffixes) for left_df, right_df in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions)]
    partition_map = {}
    for i, (_, row) in enumerate(join_pixels.iterrows()):
        pixel = HealpixPixel(order=row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME],
                             pixel=row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME])
        partition_map[pixel] = i
    meta = {}
    for name, t in left._ddf.dtypes.items():
        meta[name + suffixes[0]] = pd.Series(dtype=t)
    for name, t in right._ddf.dtypes.items():
        meta[name + suffixes[1]] = pd.Series(dtype=t)
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = HIPSCAT_ID_COLUMN
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment
