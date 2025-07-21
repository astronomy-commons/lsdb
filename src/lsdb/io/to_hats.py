from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import hats as hc
import nested_pandas as npd
import numpy as np
from hats.catalog import CatalogType, PartitionInfo
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.io.skymap import write_skymap
from hats.pixel_math import HealpixPixel, spatial_index_to_healpix
from hats.pixel_math.sparse_histogram import HistogramAggregator, SparseHistogram
from upath import UPath

from lsdb.catalog.dataset.dataset import Dataset

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


@dask.delayed
def perform_write(
    df: npd.NestedFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: str | Path | UPath,
    histogram_order: int,
    **kwargs,
) -> tuple[int, SparseHistogram]:
    """Writes a pandas dataframe to a single parquet file and returns the total count
    for the partition as well as a count histogram at the specified order.

    Args:
        df (npd.NestedFrame): dataframe to write to file
        hp_pixel (HealpixPixel): HEALPix pixel of file to be written
        base_catalog_dir (path-like): Location of the base catalog directory to write to
        histogram_order (int): Order of the count histogram
        **kwargs: other kwargs to pass to pq.write_table method

    Returns:
        The total number of points on the partition and the sparse count histogram
        at the specified order.
    """
    if len(df) == 0:
        return 0, SparseHistogram([], [], histogram_order)
    pixel_dir = hc.io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.make_directory(pixel_dir, exist_ok=True)
    pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel)
    df.to_parquet(pixel_path.path, filesystem=pixel_path.fs, **kwargs)
    return len(df), calculate_histogram(df, histogram_order)


def calculate_histogram(df: npd.NestedFrame, histogram_order: int) -> SparseHistogram:
    """Splits a partition into pixels at a specified order and computes
    the sparse histogram with the respective counts.

    Args:
        df (npd.NestedFrame): Partition data frame
        histogram_order (int): Order of the count histogram

    Returns:
        The sparse count histogram for the partition, at the specified order.
    """
    order_pixels = spatial_index_to_healpix(df.index.to_numpy(), target_order=histogram_order)
    gb = df.groupby(order_pixels, sort=False).apply(len)
    indexes, counts_at_indexes = gb.index.to_numpy(), gb.to_numpy(na_value=0)
    return SparseHistogram(indexes, counts_at_indexes, histogram_order)


# pylint: disable=protected-access,too-many-locals
def to_hats(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    catalog_name: str | None = None,
    default_columns: list[str] | None = None,
    histogram_order: int | None = None,
    overwrite: bool = False,
    create_thumbnail: bool = False,
    skymap_alt_orders: list[int] | None = None,
    addl_hats_properties: dict | None = None,
    **kwargs,
):
    """Writes a catalog to disk, in HATS format.

    The output catalog comprises  partitioned parquet files and respective metadata,
    as well as text and CSV files detailing partition, catalog and provenance info.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_path (str): Location where catalog is saved to
        catalog_name (str): The name of the output catalog
        default_columns (list[str]): A metadata property with the list of the columns in the
            catalog to be loaded by default. Uses the default columns from the original hats
            catalog if they exist.
        histogram_order (int): The default order for the count histogram. Defaults to the same
            skymap order as original catalog, or the highest order healpix of the current
            catalog data partitions.
        overwrite (bool): If True existing catalog is overwritten
        create_thumbnail (bool): If True, create a data thumbnail of the catalog for
            previewing purposes. Defaults to False.
        skymap_alt_orders (list[int]): We will write a skymap file at the ``histogram_order``,
            but can also write down-sampled skymaps, for easier previewing of the data.
        addl_hats_properties (dict): key-value pairs of additional properties to write in the
            ``hats.properties`` file.
        **kwargs: Arguments to pass to the parquet write operations
    """
    # Create the output directory for the catalog
    base_catalog_path = hc.io.file_io.get_upath(base_catalog_path)
    if hc.io.file_io.directory_has_contents(base_catalog_path):
        if not overwrite:
            raise ValueError(
                f"base_catalog_path ({str(base_catalog_path)}) contains files."
                " choose a different directory or set overwrite to True."
            )
        hc.io.file_io.remove_directory(base_catalog_path)
    hc.io.file_io.make_directory(base_catalog_path, exist_ok=True)
    if histogram_order is None:
        if catalog.hc_structure.catalog_info.skymap_order is not None:
            histogram_order = catalog.hc_structure.catalog_info.skymap_order
        else:
            max_catalog_depth = catalog.hc_structure.pixel_tree.get_max_depth()
            histogram_order = max(max_catalog_depth, 8)
    # Save partition parquet files
    pixels, counts, histograms = write_partitions(
        catalog, base_catalog_dir_fp=base_catalog_path, histogram_order=histogram_order, **kwargs
    )
    # Save parquet metadata and create a data thumbnail if needed
    hats_max_rows = max(counts)
    hc.io.write_parquet_metadata(
        base_catalog_path, create_thumbnail=create_thumbnail, thumbnail_threshold=hats_max_rows
    )
    # Save partition info
    PartitionInfo(pixels).write_to_file(base_catalog_path / "partition_info.csv")
    # Save catalog info
    if default_columns:
        missing_columns = set(default_columns) - set(catalog.columns)
        if missing_columns:
            raise ValueError(f"Default columns `{missing_columns}` not found in catalog")
    else:
        default_columns = None

    if not addl_hats_properties:
        addl_hats_properties = {}

    if catalog.hc_structure.catalog_info.catalog_type in (CatalogType.OBJECT, CatalogType.SOURCE):
        addl_hats_properties = addl_hats_properties | {
            "skymap_order": histogram_order,
            "skymap_alt_orders": skymap_alt_orders,
        }

        # Save the point distribution map
        total_histogram = HistogramAggregator(histogram_order)
        for partition_hist in histograms:
            total_histogram.add(partition_hist)
        point_map_path = hc.io.paths.get_point_map_file_pointer(base_catalog_path)
        full_histogram = total_histogram.full_histogram
        hc.io.file_io.write_fits_image(full_histogram, point_map_path)

        write_skymap(histogram=full_histogram, catalog_dir=base_catalog_path, orders=skymap_alt_orders)

    addl_hats_properties = addl_hats_properties | Dataset.new_provenance_properties(base_catalog_path)

    new_hc_structure = create_modified_catalog_structure(
        catalog.hc_structure,
        base_catalog_path,
        catalog_name if catalog_name else catalog.hc_structure.catalog_name,
        total_rows=int(np.sum(counts)),
        default_columns=default_columns,
        hats_max_rows=hats_max_rows,
        **addl_hats_properties,
    )
    new_hc_structure.catalog_info.to_properties_file(base_catalog_path)


def write_partitions(
    catalog: HealpixDataset, base_catalog_dir_fp: str | Path | UPath, histogram_order: int, **kwargs
) -> tuple[list[HealpixPixel], list[int], list[SparseHistogram]]:
    """Saves catalog partitions as parquet to disk and computes the sparse
    count histogram for each partition. The histogram is either of order 8
    or the maximum pixel order in the catalog, whichever is greater.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_dir_fp (path-like): Path to the base directory of the catalog
        histogram_order: The order of the count histogram to generate
        **kwargs: Arguments to pass to the parquet write operations

    Returns:
        A tuple with the array of non-empty pixels, the array with the total counts
        as well as the array with the sparse count histograms.
    """
    results, pixels = [], []
    partitions = catalog._ddf.to_delayed()

    for pixel, partition_index in catalog._ddf_pixel_map.items():
        results.append(
            perform_write(
                partitions[partition_index],
                pixel,
                base_catalog_dir_fp,
                histogram_order,
                **kwargs,
            )
        )
        pixels.append(pixel)

    results = dask.compute(*results)
    counts, histograms = list(zip(*results))

    non_empty_indices = np.nonzero(counts)
    non_empty_pixels = np.array(pixels)[non_empty_indices]
    non_empty_counts = np.array(counts)[non_empty_indices]
    non_empty_hists = np.array(histograms)[non_empty_indices]

    # Check that the catalog is not empty
    if len(non_empty_pixels) == 0:
        raise RuntimeError("The output catalog is empty")

    return list(non_empty_pixels), list(non_empty_counts), list(non_empty_hists)


def create_modified_catalog_structure(
    catalog_structure: HCHealpixDataset, catalog_base_dir: str | Path | UPath, catalog_name: str, **kwargs
) -> HCHealpixDataset:
    """Creates a modified version of the HATS catalog structure

    Args:
        catalog_structure (hc.catalog.Catalog): HATS catalog structure
        catalog_base_dir (UPath): Base location for the catalog
        catalog_name (str): The name of the catalog to be saved
        **kwargs: The remaining parameters to be updated in the catalog info object

    Returns:
        A HATS structure, modified with the parameters provided.
    """
    new_hc_structure = copy(catalog_structure)
    new_hc_structure.catalog_name = catalog_name
    new_hc_structure.catalog_path = catalog_base_dir
    new_hc_structure.catalog_base_dir = hc.io.file_io.get_upath(catalog_base_dir)
    new_hc_structure.on_disk = True

    new_hc_structure.catalog_info = new_hc_structure.catalog_info.copy_and_update(**kwargs)
    new_hc_structure.catalog_info.catalog_name = catalog_name
    return new_hc_structure
