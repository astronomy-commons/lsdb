from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import nested_pandas as npd
import numpy as np
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_math.pixel_margins import get_margin
from hats.pixel_math.spatial_index import healpix_to_spatial_index
from hats.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees
from hats.pixel_tree.pixel_tree import PixelTree

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import (
    AbstractCrossmatchAlgorithm,
)
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from lsdb.operations.functions.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    concat_partition_and_margin,
    filter_by_spatial_index_to_margin,
    filter_by_spatial_index_to_pixel,
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_aligned_pixels_from_alignment,
    get_healpix_pixels_from_alignment,
)
from lsdb.operations.operation import Operation

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


# pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument, too-many-locals
def perform_crossmatch(
    left_df: npd.NestedFrame,
    right_df: npd.NestedFrame,
    right_margin_df: npd.NestedFrame,
    left_margin_df: npd.NestedFrame | None,
    boundary_df: npd.NestedFrame | None,
    aligned_df: npd.NestedFrame | None,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    right_margin_pix: HealpixPixel,
    left_margin_pix: HealpixPixel,
    boundary_pixels: tuple[HealpixPixel, ...] | None,
    aligned_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    left_margin_catalog_info: TableProperties,
    boundary_catalog_info: TableProperties,
    aligned_catalog_info: TableProperties | None,
    algorithm: AbstractCrossmatchAlgorithm,
    how: str,
    suffixes: tuple[str, str],
    suffix_method: str,
    meta_df: npd.NestedFrame,
    radius_arcsec: float | None,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.

    Parameters
    ----------
    left_df : npd.NestedFrame | None
        Partition from the left catalog.
    right_df : npd.NestedFrame | None
        Partition from the right catalog.
    right_margin_df: npd.NestedFrame | None
        Partition from the right catalog margin cache.
    aligned_df : npd.NestedFrame | None
        The partition of the aligned pixel
    left_pix : HealpixPixel | None
        HealpixPixel for the left partition.
    right_pix : HealpixPixel | None
        HealpixPixel for the right partition.
    right_margin_pix : HealpixPixel | None
        HealpixPixel for the right margin partition.
    aligned_pixel : HealpixPixel
        HealpixPixel for the aligned partition.
    left_catalog_info : hats.catalog.TableProperties
        Catalog info for the left partition.
    right_catalog_info : hats.catalog.TableProperties
        Catalog info for the right partition.
    right_margin_catalog_info : hats.catalog.TableProperties
        Catalog info for the right margin partition.
    aligned_catalog_info : hats.catalog.TableProperties | None
        Catalog info for the aligned partition; usually None
    algorithm : AbstractCrossmatchAlgorithm
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how : str
        How to handle the crossmatch of the two catalogs.
        One of {'inner', 'left', 'outer'}.
    suffixes : tuple[str,str] | None
        The suffixes to append to the column names from the left and right catalogs respectively
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.
    radius_arcsec : float | None
        Matching radius of the algorithm; used to filter boundary rows to the margin
        ring of right-only pixels (outer joins only).
    left_margin_df : npd.NestedFrame | None
        Partition from the left catalog's margin cache (outer joins only).
    left_margin_pix : HealpixPixel | None
        HealpixPixel for the left margin partition (outer joins only).
    boundary_df : npd.NestedFrame | None
        Left partitions adjacent to a right-only pixel, concatenated (outer joins only).
    boundary_pixels : tuple[HealpixPixel, ...] | None
        Left pixels adjacent to a right-only aligned pixel (outer joins only).
    left_margin_catalog_info : hats.catalog.TableProperties
        Catalog info for the left margin partition (outer joins only).
    boundary_catalog_info : hats.catalog.TableProperties
        Catalog info for the boundary partitions, i.e. the left catalog (outer joins only).

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions.
    """
    # A missing left partition only yields rows for outer joins (right-only sky).
    if left_pix is not None and (left_df is None or len(left_df) == 0):
        return meta_df
    # The aligned_pixel will be the right_pix if the pixels orders are already
    # compatible, that is, it's the smaller of the left and right pixels.
    if left_pix is not None and aligned_pixel.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=left_catalog_info.healpix_order,
        )

    left_native_len: int | None = None
    if how == "outer":
        # Upstream validation guarantees a positive radius for outer joins; this task
        # never runs otherwise. The assert documents and narrows that invariant.
        assert radius_arcsec is not None
        if left_pix is None:
            # Right-only sky: the only possible left partners are rows near this pixel,
            # gathered from the left partitions adjacent to it. No left rows are native.
            if boundary_df is not None and len(boundary_df):
                left_df = filter_by_spatial_index_to_margin(
                    boundary_df,
                    aligned_pixel.order,
                    aligned_pixel.pixel,
                    radius_arcsec,
                )
            left_native_len = 0
        else:
            # Extend the left partition with its margin cache so right rows matched to
            # left rows just outside this pixel are excluded from unmatched emission.
            left_native_len = len(left_df)
            left_df = concat_partition_and_margin(left_df, left_margin_df)

    right_primary_df = right_df
    # For left/outer joins, right_df can be None - replace it with the correct empty schema.
    if right_df is None:
        # When right_df is None (partitions don't spatially overlap), we need to create
        # an empty DataFrame with the right catalog's columns and correct dtypes from meta_df.
        # The challenge: meta_df has suffixed column names, and we need to identify which
        # columns came from the right catalog and reverse the suffix transformation.
        #
        # Suffix method behavior:
        # - all_columns: All right columns have right_suffix appended
        # - overlapping_columns: Only columns present in both catalogs get suffixes;
        #   unique columns keep their original names (e.g., 'right_only' stays 'right_only')

        right_suffix = suffixes[1]

        if suffix_method == "all_columns":
            # Simple case: all right columns end with right_suffix
            right_column_names_in_meta = [col for col in meta_df.columns if col.endswith(right_suffix)]
        else:  # overlapping_columns
            # Complex case: need to identify which columns in meta came from right catalog
            # Strategy: Identify all data columns (non-extra), determine which are from left,
            # and the remainder must be from right.

            # Get algorithm extra columns (e.g., _dist_arcsec)
            extra_column_names = (
                set(algorithm.extra_columns.columns) if algorithm.extra_columns is not None else set()
            )

            # Data columns are all columns except extra columns
            data_columns = [col for col in meta_df.columns if col not in extra_column_names]

            # Identify overlapping columns: those with right_suffix came from right
            # (their left counterparts would have left_suffix)
            right_overlapping_cols = [col for col in data_columns if col.endswith(right_suffix)]

            # Determine left overlapping columns by replacing right_suffix with left_suffix
            left_suffix = suffixes[0]
            left_overlapping_base_names = [col.removesuffix(right_suffix) for col in right_overlapping_cols]
            left_overlapping_cols = [name + left_suffix for name in left_overlapping_base_names]

            # Non-overlapping columns (no suffix) could be from either left or right
            non_overlapping_cols = [
                col
                for col in data_columns
                if col not in right_overlapping_cols and col not in left_overlapping_cols
            ]

            # Separate non-overlapping: if in left_df.columns, it's from left; otherwise from right
            left_df_col_set = set(left_df.columns)
            right_non_overlapping_cols = [col for col in non_overlapping_cols if col not in left_df_col_set]

            # Combine to get all right columns in meta
            right_column_names_in_meta = right_overlapping_cols + right_non_overlapping_cols

        # Build empty right_df with right catalog's columns (removing suffixes where applied)
        def get_original_column_name(col: str) -> str:
            """Remove right_suffix if present, otherwise return as-is"""
            return col.removesuffix(right_suffix) if col.endswith(right_suffix) else col

        right_df = npd.NestedFrame(
            {
                get_original_column_name(col): pd.Series(dtype=meta_df[col].dtype)
                for col in right_column_names_in_meta
            }
        )
        # right_margin_df = right_df.copy()
    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)

    right_native_mask = np.zeros(len(right_joined_df), dtype=bool)
    if right_primary_df is not None and len(right_primary_df):
        right_native_mask[: len(right_primary_df)] = True
        if aligned_pixel.order > right_pix.order:
            spatial_index_order = right_catalog_info.healpix_order
            if spatial_index_order is None:
                raise ValueError("Right catalog must define a spatial-index order")
            lower = healpix_to_spatial_index(
                aligned_pixel.order,
                aligned_pixel.pixel,
                spatial_index_order=spatial_index_order,
            )
            upper = healpix_to_spatial_index(
                aligned_pixel.order,
                aligned_pixel.pixel + 1,
                spatial_index_order=spatial_index_order,
            )
            right_native_mask[: len(right_primary_df)] &= np.asarray(
                (right_primary_df.index >= lower) & (right_primary_df.index < upper)
            )

    crossmatch_args = CrossmatchArgs(
        left_df=left_df,
        right_df=right_joined_df,
        left_order=left_pix.order if left_pix else None,
        left_pixel=left_pix.pixel if left_pix else None,
        right_order=right_pix.order if right_pix else None,
        right_pixel=right_pix.pixel if right_pix else None,
        left_catalog_info=left_catalog_info,
        right_catalog_info=right_catalog_info,
        right_margin_catalog_info=right_margin_catalog_info,
        right_native_mask=right_native_mask,
        left_native_len=left_native_len,
    )
    return algorithm.crossmatch(crossmatch_args, how, suffixes, suffix_method)


# pylint: disable=too-many-arguments, unused-argument
def perform_crossmatch_nested(
    left_df,
    right_df,
    right_margin_df,
    aligned_df,
    left_pix,
    right_pix,
    right_margin_pix,
    aligned_pixel,
    left_catalog_info,
    right_catalog_info,
    right_margin_catalog_info,
    aligned_catalog_info,
    algorithm,
    how,
    nested_column_name,
    meta_df,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog with result in a nested column

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.

    Parameters
    ----------
    left_df : npd.NestedFrame | None
        Partition from the left catalog.
    right_df : npd.NestedFrame | None
        Partition from the right catalog.
    right_margin_df: npd.NestedFrame | None
        Partition from the right catalog margin cache.
    aligned_df : npd.NestedFrame | None
        The partition of the aligned pixel
    left_pix : HealpixPixel | None
        HealpixPixel for the left partition.
    right_pix : HealpixPixel | None
        HealpixPixel for the right partition.
    right_margin_pix : HealpixPixel | None
        HealpixPixel for the right margin partition.
    aligned_pixel : HealpixPixel
        HealpixPixel for the aligned partition.
    left_catalog_info : hats.catalog.TableProperties
        Catalog info for the left partition.
    right_catalog_info : hats.catalog.TableProperties
        Catalog info for the right partition.
    right_margin_catalog_info : hats.catalog.TableProperties
        Catalog info for the right margin partition.
    aligned_catalog_info : hats.catalog.TableProperties | None
        Catalog info for the aligned partition; usually None
    algorithm : AbstractCrossmatchAlgorithm
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how : str
        How to handle the crossmatch of the two catalogs.
        One of {'left', 'inner'}; defaults to 'inner'.
    nested_column_name : str
        The name of the nested column in the resulting dataframe storing the
        joined columns in the right catalog. (Default: name of right catalog)
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions.
        The results are stored in a nested column.
    """
    if left_df is None or len(left_df) == 0:
        return meta_df

    if aligned_pixel.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=left_catalog_info.healpix_order,
        )

    if right_df is None:
        # When right_df is None (partitions don't spatially overlap), we need to create
        # an empty DataFrame with the right catalog's columns and correct dtypes.
        # The right catalog columns are the sub-columns of the nested column in meta_df,
        # excluding any extra columns added by the algorithm (e.g., _dist_arcsec).
        extra_column_names = (
            set(algorithm.extra_columns.columns) if algorithm.extra_columns is not None else set()
        )
        nested_series = meta_df[nested_column_name]
        right_df = npd.NestedFrame(
            {
                col: pd.Series(dtype=nested_series.nest[col].dtype)
                for col in nested_series.nest.columns
                if col not in extra_column_names
            }
        )

    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)

    crossmatch_args = CrossmatchArgs(
        left_df=left_df,
        right_df=right_joined_df,
        left_order=left_pix.order,
        left_pixel=left_pix.pixel,
        right_order=right_pix.order if right_pix is not None else None,
        right_pixel=right_pix.pixel if right_pix is not None else None,
        left_catalog_info=left_catalog_info,
        right_catalog_info=right_catalog_info,
        right_margin_catalog_info=right_margin_catalog_info,
    )
    return algorithm.crossmatch_nested(crossmatch_args, nested_column_name, how)


# pylint: disable=too-many-locals
def _validate_outer_crossmatch(left: Catalog, right: Catalog, radius_arcsec: float | None):
    """Validate the margins an outer crossmatch needs to be exact.

    A right row is emitted unmatched only when no left row pairs with it anywhere, which
    requires seeing left rows across pixel boundaries: the left margin cache must reach
    at least the matching radius, and the right margin cache must reach twice the radius
    so neighborhood left rows pair against their full set of nearby right rows.
    """
    if not isinstance(radius_arcsec, (int, float)) or radius_arcsec <= 0:
        raise ValueError("how='outer' requires an algorithm with a positive 'radius_arcsec' attribute")
    if left.margin is None:
        raise ValueError("how='outer' requires the left catalog to have a margin cache")
    left_threshold = left.margin.hc_structure.catalog_info.margin_threshold
    if left_threshold is None or left_threshold < radius_arcsec:
        raise ValueError(
            f"Left margin threshold ({left_threshold}) must be at least "
            f"the matching radius ({radius_arcsec}) for how='outer'"
        )
    if right.margin is None:
        raise ValueError("how='outer' requires the right catalog to have a margin cache")
    right_threshold = right.margin.hc_structure.catalog_info.margin_threshold
    if right_threshold is None or right_threshold < 2 * radius_arcsec:
        raise ValueError(
            f"Right margin threshold ({right_threshold}) must be at least twice "
            f"the matching radius ({2 * radius_arcsec}) for how='outer'"
        )


def _boundary_pixel_lists(
    left: Catalog, pixel_mapping: pd.DataFrame
) -> list[tuple[HealpixPixel, ...] | None]:
    """For each mapping row, the left pixels adjacent to its aligned pixel.

    Right rows in sky without left coverage can only match left rows near their pixel's
    boundary; those rows live in the left partitions adjacent to it. Every left pixel
    within reach intersects one of the cells of the ring one HEALPix order finer than
    the aligned pixel, so a single alignment of all ring cells against the left pixel
    tree finds them. Returns None entries for rows that need no boundary data.
    """
    boundary_lists: list[tuple[HealpixPixel, ...] | None] = [None] * len(pixel_mapping)
    right_only_mask = pixel_mapping[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME].isna().to_numpy()
    if not right_only_mask.any():
        return boundary_lists
    rings: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ring_cells: set[tuple[int, int]] = set()
    for i in np.flatnonzero(right_only_mask):
        row = pixel_mapping.iloc[i]
        order, pixel = int(row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME]), int(
            row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME]
        )
        ring = [(order + 1, cell) for cell in get_margin(order, pixel, 1)]
        rings[(order, pixel)] = ring
        ring_cells.update(ring)
    cell_order = max(order for order, _ in ring_cells)
    intervals = np.array(
        [
            [pixel << (2 * (cell_order - order)), (pixel + 1) << (2 * (cell_order - order))]
            for order, pixel in ring_cells
        ],
        dtype=np.int64,
    )
    cell_tree = PixelTree(intervals[np.argsort(intervals[:, 0])], cell_order)
    cell_alignment = align_trees(
        cell_tree, left.hc_structure.pixel_tree, alignment_type=PixelAlignmentType.INNER
    )
    cells_to_left: dict[tuple[int, int], set[HealpixPixel]] = defaultdict(set)
    cell_mapping = cell_alignment.pixel_mapping
    for cell_order_, cell_pixel, left_order, left_pixel in zip(
        cell_mapping[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME],
        cell_mapping[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME],
        cell_mapping[PixelAlignment.JOIN_ORDER_COLUMN_NAME],
        cell_mapping[PixelAlignment.JOIN_PIXEL_COLUMN_NAME],
    ):
        cells_to_left[(int(cell_order_), int(cell_pixel))].add(HealpixPixel(int(left_order), int(left_pixel)))
    for i in np.flatnonzero(right_only_mask):
        row = pixel_mapping.iloc[i]
        key = (
            int(row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME]),
            int(row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME]),
        )
        adjacent: set[HealpixPixel] = set()
        for cell in rings[key]:
            adjacent |= cells_to_left.get(cell, set())
        boundary_lists[i] = tuple(sorted(adjacent))
    return boundary_lists


def _plan_outer_alignment(
    left: Catalog, right: Catalog, alignment: PixelAlignment
) -> tuple[PixelAlignment, list[tuple[HealpixPixel, ...] | None]]:
    """Adjust an OUTER alignment for crossmatching.

    Drops aligned pixels that exist only in the right margin cache's halo: they hold no
    primary right rows and would only create empty partitions, then plans the boundary
    partitions each right-only pixel needs (see ``_boundary_pixel_lists``).
    """
    pixel_mapping = alignment.pixel_mapping
    right_only = pixel_mapping[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME].isna()
    if right_only.any():
        primary_right_pixels = {(p.order, p.pixel) for p in right.get_healpix_pixels()}
        join_keys = pd.Series(
            list(
                zip(
                    pixel_mapping[PixelAlignment.JOIN_ORDER_COLUMN_NAME],
                    pixel_mapping[PixelAlignment.JOIN_PIXEL_COLUMN_NAME],
                )
            ),
            index=pixel_mapping.index,
        )
        pixel_mapping = pixel_mapping[~right_only | join_keys.isin(primary_right_pixels)].reset_index(
            drop=True
        )
    tree_order = alignment.pixel_tree.tree_order
    if len(pixel_mapping):
        orders = pixel_mapping[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME].to_numpy(dtype=np.int64)
        pixels = pixel_mapping[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME].to_numpy(dtype=np.int64)
        shift = 2 * (tree_order - orders)
        intervals = np.stack([np.left_shift(pixels, shift), np.left_shift(pixels + 1, shift)], axis=1).astype(
            np.int64
        )
    else:
        intervals = np.empty((0, 2), dtype=np.int64)
    alignment = PixelAlignment(
        PixelTree(intervals, tree_order), pixel_mapping, alignment.alignment_type, alignment.moc
    )
    return alignment, _boundary_pixel_lists(left, pixel_mapping)


def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    algorithm: AbstractCrossmatchAlgorithm,
    how: str,
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
    log_changes: bool = True,
) -> tuple[Operation, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to perform the cross-match on
    right : lsdb.Catalog
        The right catalog to perform the cross-match on
    algorithm : AbstractCrossmatchAlgorithm, default `KDTreeCrossmatch`
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how: str
        How to handle the crossmatch of the two catalogs.
        One of {'inner', 'left', 'outer'}.
    suffixes : tuple[str,str]
        The suffixes to append to the column names from the left and
        right catalogs respectively.
    suffix_method : str | None, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'. Default: True

    Returns
    -------
    tuple[Operation, PixelAlignment]
        A tuple of the LSDB Operation with the result of the cross-match,
        and the PixelAlignment of the two input catalogs.
    """
    # Validate the algorithm parameters
    algorithm.validate(left, right)

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    # ``outer`` aligns pixels in the outer way: every pixel of either catalog is visited,
    # so right rows in sky without left coverage are emitted unmatched. The right margin
    # halo is not added for outer joins: it would only inflate the result catalog's MOC,
    # and OUTER alignment already keeps every left pixel.
    alignment = align_catalogs(
        left, right, add_right_margin=how != "outer", alignment_type=PixelAlignmentType[how.upper()]
    )
    radius_arcsec = getattr(algorithm, "radius_arcsec", None)
    boundary_pixel_lists: list[tuple[HealpixPixel, ...] | None] = []
    if how == "outer":
        _validate_outer_crossmatch(left, right, radius_arcsec)
        alignment, boundary_pixel_lists = _plan_outer_alignment(left, right, alignment)
    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_joined_tables(
        (left, right),
        suffixes,
        suffix_method=suffix_method,
        extra_columns=algorithm.extra_columns,
        log_changes=log_changes,
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    empty_pixels: list[HealpixPixel | None] = [None] * len(aligned_pixels)
    op = align_and_apply(
        [
            (left, left_pixels),
            (right, right_pixels),
            (right.margin, right_pixels),
            (left.margin if how == "outer" else None, left_pixels if how == "outer" else empty_pixels),
            (left if how == "outer" else None, boundary_pixel_lists or empty_pixels),
            (None, aligned_pixels),
        ],
        perform_crossmatch,
        meta_df,
        aligned_pixels,
        algorithm,
        how,
        suffixes,
        suffix_method,
        meta_df,
        radius_arcsec,
    )

    return op, alignment


# pylint: disable=too-many-locals
def crossmatch_catalog_data_nested(
    left: Catalog,
    right: Catalog,
    algorithm: AbstractCrossmatchAlgorithm,
    how: str,
    nested_column_name: str,
) -> tuple[Operation, PixelAlignment]:
    """Crossmatches the data from two catalogs with the result from the right catalog in a nested column

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to perform the cross-match on
    right : lsdb.Catalog
        The right catalog to perform the cross-match on
    algorithm : AbstractCrossmatchAlgorithm, default `KDTreeCrossmatch`
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how : str
        How to handle the crossmatch of the two catalogs.
        One of {'left', 'inner'}.
    nested_column_name : str
        The name of the nested column that will contain the crossmatched rows
        from the right catalog

    Returns
    -------
    tuple[Operation, PixelAlignment]
        A tuple of the LSDB Operation with the result of the cross-match,
        and the PixelAlignment of the two input catalogs.
    """
    # Validate the algorithm parameters
    algorithm.validate(left, right)

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    # perform alignment on the two catalogs
    alignment = align_catalogs(
        left, right, add_right_margin=True, alignment_type=PixelAlignmentType[how.upper()]
    )

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_nested_tables(
        [left], right, nested_column_name, extra_nested_columns=algorithm.extra_columns
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    op = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels), (None, aligned_pixels)],
        perform_crossmatch_nested,
        meta_df,
        aligned_pixels,
        algorithm,
        how,
        nested_column_name,
        meta_df,
    )

    return op, alignment
