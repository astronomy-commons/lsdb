from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hats as hc
import numpy as np
from hats.catalog import CatalogType, PartitionInfo
from hats.io.skymap import write_skymap
from hats.pixel_math.sparse_histogram import HistogramAggregator
from upath import UPath

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.io.to_hats import create_modified_catalog_structure, write_partitions

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


# pylint: disable=protected-access,too-many-locals, duplicate-code
def to_map(
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
    """Special case to write a MAP catalog to disk.

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

    catalog.hc_structure.catalog_info.catalog_type = CatalogType.MAP
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
