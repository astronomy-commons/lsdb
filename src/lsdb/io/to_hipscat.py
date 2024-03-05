from __future__ import annotations

import dataclasses
from copy import copy
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, Union

import dask
import hipscat as hc
import pandas as pd
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.io import FilePointer
from hipscat.pixel_math import HealpixPixel

from lsdb.types import HealpixInfo

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


@dask.delayed
def perform_write(
    df: pd.DataFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: FilePointer,
    storage_options: dict | None = None,
    **kwargs,
) -> int:
    """Performs a write of a pandas dataframe to a single parquet file, following the hipscat structure.

    To be used as a dask delayed method as part of a dask task graph.

    Args:
        df (pd.DataFrame): dataframe to write to file
        hp_pixel (HealpixPixel): HEALPix pixel of file to be written
        base_catalog_dir (FilePointer): Location of the base catalog directory to write to
        storage_options (dict): fsspec storage options
        **kwargs: other kwargs to pass to pd.to_parquet method

    Returns:
        number of rows written to disk
    """
    if len(df) == 0:
        return 0
    pixel_dir = hc.io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.make_directory(pixel_dir, exist_ok=True, storage_options=storage_options)
    pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.write_dataframe_to_parquet(df, pixel_path, storage_options, **kwargs)
    return len(df)


# pylint: disable=W0212
def to_hipscat(
    catalog: HealpixDataset,
    base_catalog_path: str,
    catalog_name: Union[str, None] = None,
    overwrite: bool = False,
    storage_options: dict | None = None,
    **kwargs,
):
    """Writes a catalog to disk, in HiPSCat format. The output catalog comprises
    partition parquet files and respective metadata, as well as JSON files detailing
    partition, catalog and provenance info.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_path (str): Location where catalog is saved to
        catalog_name (str): The name of the output catalog
        overwrite (bool): If True existing catalog is overwritten
        storage_options (dict): Dictionary that contains abstract filesystem credentials
        **kwargs: Arguments to pass to the parquet write operations
    """
    # Create base directory
    base_catalog_dir_fp = hc.io.get_file_pointer_from_path(base_catalog_path)
    hc.io.file_io.make_directory(base_catalog_dir_fp, overwrite, storage_options)
    # Save partition parquet files
    pixel_to_partition_size_map = write_partitions(catalog, base_catalog_dir_fp, storage_options, **kwargs)
    # Save parquet metadata
    hc.io.write_parquet_metadata(base_catalog_path, storage_options, **kwargs)
    # Save partition info
    partition_info = _get_partition_info_dict(pixel_to_partition_size_map)
    hc.io.write_partition_info(base_catalog_dir_fp, partition_info, storage_options)
    # Save catalog info
    new_hc_structure = create_modified_catalog_structure(
        catalog.hc_structure,
        base_catalog_path,
        catalog_name if catalog_name else catalog.hc_structure.catalog_name,
        total_rows=sum(pi[0] for pi in partition_info.values()),
    )
    hc.io.write_catalog_info(
        catalog_base_dir=base_catalog_path,
        dataset_info=new_hc_structure.catalog_info,
        storage_options=storage_options,
    )
    # Save provenance info
    hc.io.write_metadata.write_provenance_info(
        catalog_base_dir=base_catalog_dir_fp,
        dataset_info=new_hc_structure.catalog_info,
        tool_args=_get_provenance_info(new_hc_structure),
        storage_options=storage_options,
    )


def write_partitions(
    catalog: HealpixDataset,
    base_catalog_dir_fp: FilePointer,
    storage_options: Union[Dict[Any, Any], None] = None,
    **kwargs,
) -> Dict[HealpixPixel, int]:
    """Saves catalog partitions as parquet to disk

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_dir_fp (FilePointer): Path to the base directory of the catalog
        storage_options (dict): Dictionary that contains abstract filesystem credentials
        **kwargs: Arguments to pass to the parquet write operations

    Returns:
        A dictionary mapping each HEALPix pixel to the number of data points in it.
    """
    results = []
    pixel_to_result_index = {}

    partitions = catalog._ddf.to_delayed()

    for index, (pixel, partition_index) in enumerate(catalog._ddf_pixel_map.items()):
        results.append(
            perform_write(
                partitions[partition_index],
                pixel,
                base_catalog_dir_fp,
                storage_options,
                **kwargs,
            )
        )
        pixel_to_result_index[pixel] = index

    partition_sizes = dask.compute(*results)

    if all(size == 0 for size in partition_sizes):
        raise RuntimeError("The output catalog is empty")

    pixel_to_partition_size_map = {
        pixel: partition_sizes[index]
        for pixel, index in pixel_to_result_index.items()
        if partition_sizes[index] > 0
    }

    return pixel_to_partition_size_map


def _get_partition_info_dict(ddf_points_map: Dict[HealpixPixel, int]) -> Dict[HealpixPixel, HealpixInfo]:
    """Creates the partition info dictionary

    Args:
        ddf_points_map (Dict[HealpixPix,int]): Dictionary mapping each HealpixPixel
            to the respective number of points inside its partition

    Returns:
        A partition info dictionary, where the keys are the HEALPix pixels and
        the values are pairs where the first element is the number of points
        inside the pixel, and the second is the list of destination pixel numbers.
    """
    return {pixel: (length, [pixel.pixel]) for pixel, length in ddf_points_map.items()}


def create_modified_catalog_structure(
    catalog_structure: HCHealpixDataset, catalog_base_dir: str, catalog_name: str, **kwargs
) -> HCHealpixDataset:
    """Creates a modified version of the HiPSCat catalog structure

    Args:
        catalog_structure (hc.catalog.Catalog): HiPSCat catalog structure
        catalog_base_dir (str): Base location for the catalog
        catalog_name (str): The name of the catalog to be saved
        **kwargs: The remaining parameters to be updated in the catalog info object

    Returns:
        A HiPSCat structure, modified with the parameters provided.
    """
    new_hc_structure = copy(catalog_structure)
    new_hc_structure.catalog_name = catalog_name
    new_hc_structure.catalog_path = catalog_base_dir
    new_hc_structure.catalog_base_dir = hc.io.file_io.get_file_pointer_from_path(catalog_base_dir)
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
