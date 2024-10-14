from __future__ import annotations

import dataclasses
from copy import copy
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import dask
import hipscat as hc
import hipscat.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.pixel_math import HealpixPixel, hipscat_id_to_healpix
from hipscat_import.catalog.sparse_histogram import SparseHistogram
from upath import UPath

from lsdb.types import HealpixInfo

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


@dask.delayed
def perform_write(
    partition: npd.NestedFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: str | Path | UPath,
    *,
    histogram_order: int,
    **kwargs,
) -> np.ndarray:
    """Writes a pandas dataframe to a single parquet file and returns the point
    distribution map (count histogram) at the specified order.

    Args:
        partition (npd.NestedFrame): Partition data frame to write to file
        hp_pixel (HealpixPixel): HEALPix pixel of file to be written
        base_catalog_dir (path-like): Location of the base catalog directory to write to
        histogram_order (int): Order of the count histogram
        **kwargs: other kwargs to pass to pd.to_parquet method

    Returns:
        The sparse count histogram for the partition, at the specified order.
    """
    if len(partition) == 0:
        return SparseHistogram.make_empty(healpix_order=histogram_order)
    pixel_dir = hc.io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.make_directory(pixel_dir, exist_ok=True)
    pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel)
    hc.io.file_io.write_dataframe_to_parquet(partition, pixel_path, **kwargs)
    return calculate_histogram(partition, histogram_order)


def calculate_histogram(partition: npd.NestedFrame, histogram_order: int) -> np.ndarray:
    """Splits a partition into pixels at a specified order and computes
    the sparse histogram with the respective counts.

    Args:
        partition (npd.NestedFrame): Partition data frame
        histogram_order (int): Order of the count histogram

    Returns:
        The sparse count histogram for the partition, at the specified order.
    """
    order_pixels = hipscat_id_to_healpix(partition.index.to_numpy(), target_order=histogram_order)
    gb = partition.groupby(order_pixels, sort=False).apply(len)
    indexes, counts_at_indexes = gb.index.to_numpy(), gb.to_numpy(na_value=0)
    return SparseHistogram.make_from_counts(indexes, counts_at_indexes, histogram_order)


# pylint: disable=W0212
def to_hipscat(
    catalog: HealpixDataset,
    base_catalog_path: str | Path | UPath,
    catalog_name: Union[str, None] = None,
    overwrite: bool = False,
    **kwargs,
):
    """Writes a catalog to disk, in HiPSCat format.

    The output catalog includes partition parquet files and respective metadata,
    JSON files detailing partition, catalog and provenance info, as well as a
    point distribution map in FITS format. The latter consists of a histogram of
    of order 8 or the maximum pixel order in the catalog, whichever is greater.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_path (str): Location where catalog is saved to
        catalog_name (str): The name of the output catalog
        overwrite (bool): If True existing catalog is overwritten
        **kwargs: Arguments to pass to the parquet write operations
    """
    # Create the output directory for the catalog
    if hc.io.file_io.directory_has_contents(base_catalog_path):
        if not overwrite:
            raise ValueError(
                f"base_catalog_path ({str(base_catalog_path)}) contains files."
                " choose a different directory or set overwrite to True."
            )
        hc.io.file_io.remove_directory(base_catalog_path)
    hc.io.file_io.make_directory(base_catalog_path, exist_ok=True)

    # Save partition parquet files
    pixel_to_partition_hist_map = write_partitions(catalog, base_catalog_path, **kwargs)

    # Save parquet metadata
    hc.io.write_parquet_metadata(base_catalog_path)

    # Save partition info
    partition_info = _get_partition_info_dict(pixel_to_partition_hist_map)
    hc.io.write_partition_info(base_catalog_path, partition_info)

    # Save catalog info
    new_hc_structure = create_modified_catalog_structure(
        catalog.hc_structure,
        base_catalog_path,
        catalog_name if catalog_name else catalog.hc_structure.catalog_name,
        total_rows=sum(pi[0] for pi in partition_info.values()),
    )
    hc.io.write_catalog_info(catalog_base_dir=base_catalog_path, dataset_info=new_hc_structure.catalog_info)

    # Save provenance info
    hc.io.write_metadata.write_provenance_info(
        catalog_base_dir=base_catalog_path,
        dataset_info=new_hc_structure.catalog_info,
        tool_args=_get_provenance_info(new_hc_structure),
    )

    # Save the point distribution map
    sparse_hist = sum(hist.sparse_array for hist in pixel_to_partition_hist_map.values())
    dense_histogram = sparse_hist.toarray()[0]
    hc.io.write_metadata.write_fits_map(base_catalog_path, dense_histogram)


def write_partitions(
    catalog: HealpixDataset, base_catalog_dir_fp: str | Path | UPath, **kwargs
) -> Dict[HealpixPixel, SparseHistogram]:
    """Saves catalog partitions as parquet to disk and computes the sparse
    count histogram for each partition. The histogram is either of order 8
    or the maximum pixel order in the catalog, whichever is greater.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_dir_fp (path-like): Path to the base directory of the catalog
        **kwargs: Arguments to pass to the parquet write operations

    Returns:
        A dictionary mapping each HEALPix pixel to the sparse count histogram.
    """
    default_histogram_order = hp.nside2order(256)
    histogram_order = max(catalog.hc_structure.pixel_tree.get_max_depth(), default_histogram_order)

    results = []
    pixel_to_result_index = {}
    partitions = catalog._ddf.to_delayed()

    for index, (pixel, partition_index) in enumerate(catalog._ddf_pixel_map.items()):
        results.append(
            perform_write(
                partitions[partition_index],
                pixel,
                base_catalog_dir_fp,
                histogram_order=histogram_order,
                **kwargs,
            )
        )
        pixel_to_result_index[pixel] = index

    partitions_hist = dask.compute(*results)

    pixel_to_partition_hist_map = {
        pixel: partitions_hist[index]
        for pixel, index in pixel_to_result_index.items()
        if partitions_hist[index].sparse_array.sum() > 0
    }

    if len(pixel_to_partition_hist_map) == 0:
        raise RuntimeError("The output catalog is empty")

    return pixel_to_partition_hist_map


def _get_partition_info_dict(
    pixel_to_partition_size_map: Dict[HealpixPixel, SparseHistogram]
) -> Dict[HealpixPixel, HealpixInfo]:
    """Creates the partition info dictionary

    Args:
        pixel_to_partition_size_map (Dict[HealpixPix,SparseHistogram]): Dictionary
            mapping each HEALPix pixel to the respective count histogram.

    Returns:
        A partition info dictionary, where the keys are the HEALPix pixels and
        the values are pairs where the first element is the number of points
        inside the pixel, and the second is the list of destination pixel numbers.
    """
    return {
        pixel: (count_hist.sparse_array.sum(), [pixel.pixel])
        for pixel, count_hist in pixel_to_partition_size_map.items()
    }


def create_modified_catalog_structure(
    catalog_structure: HCHealpixDataset, catalog_base_dir: str | Path | UPath, catalog_name: str, **kwargs
) -> HCHealpixDataset:
    """Creates a modified version of the HiPSCat catalog structure

    Args:
        catalog_structure (hc.catalog.Catalog): HiPSCat catalog structure
        catalog_base_dir (UPath): Base location for the catalog
        catalog_name (str): The name of the catalog to be saved
        **kwargs: The remaining parameters to be updated in the catalog info object

    Returns:
        A HiPSCat structure, modified with the parameters provided.
    """
    new_hc_structure = copy(catalog_structure)
    new_hc_structure.catalog_name = catalog_name
    new_hc_structure.catalog_path = catalog_base_dir
    new_hc_structure.catalog_base_dir = hc.io.file_io.get_upath(catalog_base_dir)
    new_hc_structure.on_disk = True
    new_hc_structure.catalog_info = dataclasses.replace(
        new_hc_structure.catalog_info, catalog_name=catalog_name, **kwargs
    )
    return new_hc_structure


def _get_provenance_info(catalog_structure: HCHealpixDataset) -> dict:
    """Fill all known information in a dictionary for provenance tracking.

    Args:
        catalog_structure (HCHealpixDataset): The catalog structure

    Returns:
        dictionary with all argument_name -> argument_value as key -> value pairs.
    """
    structure_args = {
        "catalog_name": catalog_structure.catalog_name,
        "output_path": catalog_structure.catalog_path,
        "output_catalog_name": catalog_structure.catalog_name,
        "catalog_path": catalog_structure.catalog_path,
    }
    args = {
        **structure_args,
        **dataclasses.asdict(catalog_structure.catalog_info),
    }
    provenance_info = {
        "tool_name": "lsdb",
        "version": version("lsdb"),
        "runtime_args": args,
    }
    return provenance_info
