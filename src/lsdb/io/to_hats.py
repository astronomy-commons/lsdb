from __future__ import annotations

import dataclasses
from copy import copy
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import dask
import hats as hc
import nested_pandas as npd
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math import HealpixPixel
from upath import UPath

from lsdb.types import HealpixInfo

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


@dask.delayed
def perform_write(
    df: npd.NestedFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: str | Path | UPath,
    **kwargs,
) -> int:
    """Performs a write of a pandas dataframe to a single parquet file, following the hats structure.

    To be used as a dask delayed method as part of a dask task graph.

    Args:
        df (npd.NestedFrame): dataframe to write to file
        hp_pixel (HealpixPixel): HEALPix pixel of file to be written
        base_catalog_dir (path-like): Location of the base catalog directory to write to
        **kwargs: other kwargs to pass to pd.to_parquet method

    Returns:
        number of rows written to disk
    """
    if len(df) == 0:
        return 0
    pixel_dir = hc.io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.make_directory(pixel_dir, exist_ok=True)
    pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel)
    hc.io.file_io.write_dataframe_to_parquet(df, pixel_path, **kwargs)
    return len(df)


# pylint: disable=W0212
def to_hats(
    catalog: HealpixDataset,
    base_catalog_path: str | Path | UPath,
    catalog_name: Union[str, None] = None,
    overwrite: bool = False,
    **kwargs,
):
    """Writes a catalog to disk, in HATS format. The output catalog comprises
    partition parquet files and respective metadata, as well as JSON files detailing
    partition, catalog and provenance info.

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
    pixel_to_partition_size_map = write_partitions(catalog, base_catalog_path, **kwargs)
    # Save parquet metadata
    hc.io.write_parquet_metadata(base_catalog_path)
    # Save partition info
    partition_info = _get_partition_info_dict(pixel_to_partition_size_map)
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


def write_partitions(
    catalog: HealpixDataset, base_catalog_dir_fp: str | Path | UPath, **kwargs
) -> Dict[HealpixPixel, int]:
    """Saves catalog partitions as parquet to disk

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_dir_fp (UPath): Path to the base directory of the catalog
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