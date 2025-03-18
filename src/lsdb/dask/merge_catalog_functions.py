from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import hats.pixel_math.healpix_shim as hp
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask.dataframe.dispatch import make_meta
from dask.delayed import Delayed, delayed
from hats.catalog import TableProperties
from hats.io import paths
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, healpix_to_spatial_index
from hats.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees
from hats.pixel_tree.moc_utils import copy_moc
from hats.pixel_tree.pixel_alignment import align_with_mocs

from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def concat_partition_and_margin(
    partition: npd.NestedFrame, margin: npd.NestedFrame | None
) -> npd.NestedFrame:
    """Concatenates a partition and margin dataframe together

    Args:
        partition (npd.NestedFrame): The partition dataframe
        margin (npd.NestedFrame): The margin dataframe

    Returns:
        The concatenated dataframe with the partition on top and the margin on the bottom
    """
    if margin is None:
        return partition

    joined_df = pd.concat([partition, margin])
    return npd.NestedFrame(joined_df)


def remove_hips_columns(df: npd.NestedFrame | None):
    """Removes any HIPS Norder, Dir, and Npix columns from a dataframe

    Args:
        df (npd.NestedFrame): The catalog dataframe

    Returns:
        The dataframe with the columns removed
    """
    if df is None:
        return None
    hive_columns_in_df = [c for c in paths.HIVE_COLUMNS if c in df.columns]
    return df.drop(columns=hive_columns_in_df)


def align_catalogs(left: Catalog, right: Catalog, add_right_margin: bool = True) -> PixelAlignment:
    """Aligns two catalogs, also using the right catalog's margin if it exists

    Args:
        left (lsdb.Catalog): The left catalog to align
        right (lsdb.Catalog): The right catalog to align
        add_right_margin (bool): If True, when using MOCs to align catalogs, adds a border to the
            right catalog's moc to include the margin of the right catalog, if it exists. Defaults to True.
    Returns:
        The PixelAlignment object from aligning the catalogs
    """

    right_added_radius = None

    if right.margin is not None:
        right_tree = align_trees(
            right.hc_structure.pixel_tree,
            right.margin.hc_structure.pixel_tree,
            alignment_type=PixelAlignmentType.OUTER,
        ).pixel_tree
        if add_right_margin:
            right_added_radius = right.margin.hc_structure.catalog_info.margin_threshold
    else:
        right_tree = right.hc_structure.pixel_tree

    right_moc = (
        right.hc_structure.moc
        if right.hc_structure.moc is not None
        else right.hc_structure.pixel_tree.to_moc()
    )
    if right_added_radius is not None:
        right_moc_depth_resol = hp.order2resol(right_moc.max_order, arcmin=True) * 60
        if right_added_radius < right_moc_depth_resol:
            right_moc = copy_moc(right_moc).add_neighbours()
        else:
            delta_order = int(np.ceil(np.log2(right_added_radius / right_moc_depth_resol)))
            right_moc = right_moc.degrade_to_order(right_moc.max_order - delta_order).add_neighbours()

    return align_with_mocs(
        left.hc_structure.pixel_tree,
        right_tree,
        left.hc_structure.moc,
        right_moc,
        alignment_type=PixelAlignmentType.INNER,
    )


def align_and_apply(
    catalog_mappings: list[tuple[HealpixDataset | None, list[HealpixPixel]]], func: Callable, *args, **kwargs
) -> list[Delayed]:
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

    # gets the pixels and hc_structures to pass to the function
    pixels = [pixels for (_, pixels) in catalog_mappings]
    for p in pixels:
        if len(p) == 0:
            raise RuntimeError("Catalogs do not overlap")

    catalog_infos = [
        cat.hc_structure.catalog_info if cat is not None else None for (cat, _) in catalog_mappings
    ]

    # aligns the catalog's partitions to the given pixels for each catalog
    aligned_partitions = [align_catalog_to_partitions(cat, pixels) for (cat, pixels) in catalog_mappings]

    # defines an inner function that can be vectorized to apply the given function to each of the partitions
    # with the additional arguments including as the hc_structures and any specified additional arguments
    def apply_func(*partitions_and_pixels):
        return perform_align_and_apply_func(
            len(aligned_partitions), func, *partitions_and_pixels, *catalog_infos, *args, **kwargs
        )

    resulting_partitions = np.vectorize(apply_func)(*aligned_partitions, *pixels)
    return resulting_partitions


@delayed
def perform_align_and_apply_func(num_partitions, func, *args, **kwargs):
    """Performs the function inside `align_and_apply` and updates hive columns"""
    filtered_parts = []
    partitions = args[:num_partitions]
    pixels = args[num_partitions : 2 * num_partitions]
    for df in partitions:
        filtered_parts.append(remove_hips_columns(df))
    return func(
        *filtered_parts,
        *pixels,
        *args[2 * num_partitions :],
        **kwargs,
    )


def filter_by_spatial_index_to_pixel(dataframe: npd.NestedFrame, order: int, pixel: int) -> npd.NestedFrame:
    """Filters a catalog dataframe to the points within a specified HEALPix pixel using the spatial index

    Args:
        dataframe (npd.NestedFrame): The dataframe to filter
        order (int): The order of the HEALPix pixel to filter to
        pixel (int): The pixel number in NESTED numbering of the HEALPix pixel to filter to

    Returns:
        The filtered dataframe with only the rows that are within the specified HEALPix pixel
    """
    lower_bound = healpix_to_spatial_index(order, pixel)
    upper_bound = healpix_to_spatial_index(order, pixel + 1)
    filtered_df = dataframe[(dataframe.index >= lower_bound) & (dataframe.index < upper_bound)]
    return filtered_df


def construct_catalog_args(
    partitions: list[Delayed], meta_df: npd.NestedFrame, alignment: PixelAlignment
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Constructs the arguments needed to create a catalog from a list of delayed partitions

    Args:
        partitions (List[Delayed]): The list of delayed partitions to create the catalog from
        meta_df (npd.NestedFrame): The dask meta schema for the partitions
        alignment (PixelAlignment): The alignment used to create the delayed partitions

    Returns:
        A tuple of (ddf, partition_map, alignment) with the dask dataframe, the partition map, and the
        alignment needed to create the catalog
    """
    # generate dask df partition map from alignment
    partition_map = get_partition_map_from_alignment_pixels(alignment.pixel_mapping)
    # create dask df from delayed partitions
    divisions = get_pixels_divisions(list(partition_map.keys()))
    partitions = partitions if len(partitions) > 0 else [delayed(meta_df.copy())]
    ddf = nd.NestedFrame.from_delayed(partitions, meta=meta_df, divisions=divisions, verify_meta=True)
    return ddf, partition_map, alignment


def get_healpix_pixels_from_alignment(
    alignment: PixelAlignment,
) -> tuple[list[HealpixPixel], list[HealpixPixel]]:
    """Gets the list of primary and join pixels as the HealpixPixel class from a PixelAlignment

    Args:
        alignment (PixelAlignment): the PixelAlignment to get pixels from

    Returns:
        a tuple of (primary_pixels, join_pixels) with lists of HealpixPixel objects
    """
    pixel_mapping = alignment.pixel_mapping
    if len(pixel_mapping) == 0:
        return ([], [])
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
    index_name: str = SPATIAL_INDEX_COLUMN,
    index_type: npt.DTypeLike | None = None,
) -> npd.NestedFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs

    Creates an empty dataframe with the columns of each catalog appended with a suffix. Allows specifying
    extra columns that should also be added, and the name of the index of the resulting dataframe.

    Args:
        catalogs (Sequence[lsdb.Catalog]): The catalogs to merge together
        suffixes (Sequence[Str]): The column suffixes to apply each catalog
        extra_columns (pd.Dataframe): Any additional columns to the merged catalogs
        index_name (str): The name of the index in the resulting DataFrame
        index_type (npt.DTypeLike): The type of the index in the resulting DataFrame.
            Default: type of index in the first catalog

    Returns:
        An empty dataframe with the columns of each catalog with their respective suffix, and any extra
        columns specified, with the index name set.
    """
    meta = {}
    # Construct meta for crossmatched catalog columns
    for table, suffix in zip(catalogs, suffixes):
        for name, col_type in table.dtypes.items():
            if name not in paths.HIVE_COLUMNS:
                meta[name + suffix] = pd.Series(dtype=col_type)
    # Construct meta for crossmatch result columns
    if extra_columns is not None:
        meta.update(extra_columns)
    if index_type is None:
        # pylint: disable=protected-access
        index_type = catalogs[0]._ddf._meta.index.dtype
    index = pd.Index(pd.Series(dtype=index_type), name=index_name)
    meta_df = npd.NestedFrame(pd.DataFrame(meta, index))
    return meta_df


def generate_meta_df_for_nested_tables(
    catalogs: Sequence[Catalog],
    nested_catalog: Catalog,
    nested_column_name: str,
    join_column_name: str,
    extra_columns: pd.DataFrame | None = None,
    index_name: str = SPATIAL_INDEX_COLUMN,
    index_type: npt.DTypeLike | None = None,
) -> npd.NestedFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs, adding the right as a
    nested frame

    Creates an empty dataframe with the columns of the left catalog, and a nested column with the right
    catalog. Allows specifying extra columns that should also be added, and the name of the index of the
    resulting dataframe.

    Args:
        catalogs (Sequence[lsdb.Catalog]): The catalogs to merge together
        nested_catalog (Catalog): The catalog to add as a nested column
        nested_column_name (str): The name of the nested column
        join_column_name (str): The name of the column in the right catalog to join on
        extra_columns (pd.Dataframe): Any additional columns to the merged catalogs
        index_name (str): The name of the index in the resulting DataFrame
        index_type (npt.DTypeLike): The type of the index in the resulting DataFrame

    Returns:
        An empty dataframe with the right catalog joined to the left as a nested column, and any extra
        columns specified, with the index name set.
    """
    meta = {}
    # Construct meta for crossmatched catalog columns
    for table in catalogs:
        for name, col_type in table.dtypes.items():
            if name not in paths.HIVE_COLUMNS:
                meta[name] = pd.Series(dtype=col_type)
    # Construct meta for crossmatch result columns
    if extra_columns is not None:
        meta.update(extra_columns)

    if index_type is None:
        # pylint: disable=protected-access
        index_type = catalogs[0]._ddf._meta.index.dtype
    index = pd.Index(pd.Series(dtype=index_type), name=index_name)
    meta_df = pd.DataFrame(meta, index)

    # make an empty copy of the nested catalog, removing the column that will be joined on (and removed from
    # the eventual dataframe)
    # pylint: disable=protected-access
    nested_catalog_meta = nested_catalog._ddf._meta.copy().iloc[:0].drop(join_column_name, axis=1)
    hive_cols_to_drop = [c for c in paths.HIVE_COLUMNS if c in nested_catalog_meta.columns]
    nested_catalog_meta = nested_catalog_meta.drop(columns=hive_cols_to_drop)

    meta_df = npd.NestedFrame(meta_df).add_nested(nested_catalog_meta, nested_column_name)

    # Use nested-pandas to make the resulting meta with the nested catalog meta as a nested column
    return meta_df


def concat_metas(metas: Sequence[npd.NestedFrame | dict]):
    """Concats the columns of a sequence of dask metas into a single NestedFrame meta

    Args:
        metas (Sequence[dict | DataFrame]): A collection of dask meta inputs

    Returns:
        (npd.NestedFrame) An empty NestedFrame with the columns of the input metas concatenated together in
        the order of the input sequence.
    """
    pandas_metas = []
    for meta in metas:
        pandas_metas.append(npd.NestedFrame(make_meta(meta)))
    return npd.NestedFrame(pd.concat(pandas_metas, axis=1))


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
    catalog: HealpixDataset | None, pixels: list[HealpixPixel]
) -> list[Delayed | None]:
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


def create_merged_catalog_info(
    left_info: TableProperties, right_info: TableProperties, updated_name: str, suffixes: tuple[str, str]
) -> TableProperties:
    """Creates the catalog info of the resulting catalog from merging two catalogs

    Updates the ra and dec columns names, and any default columns by adding the correct suffixes, updates the
    catalog name, and sets the total rows to 0

    Args:
        left_info (TableProperties): The catalog_info of the left catalog
        right_info (TableProperties): The catalog_info of the right catalog
        updated_name (str): The updated name of the catalog
        suffixes (tuple[str, str]): The suffixes of the catalogs in the merged result
    """
    default_cols = (
        [c + suffixes[0] for c in left_info.default_columns] if left_info.default_columns is not None else []
    )
    default_cols = (
        default_cols + [c + suffixes[1] for c in right_info.default_columns]
        if right_info.default_columns is not None
        else default_cols
    )
    default_cols_to_use = default_cols if len(default_cols) > 0 else None
    return left_info.copy_and_update(
        catalog_name=updated_name,
        ra_column=left_info.ra_column + suffixes[0],
        dec_column=left_info.dec_column + suffixes[0],
        total_rows=0,
        default_columns=default_cols_to_use,
    )
