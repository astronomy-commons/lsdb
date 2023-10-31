import dataclasses
import os
from typing import TYPE_CHECKING

import dask
import hipscat.io.paths
import pandas as pd
from hipscat.pixel_math import HealpixPixel

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


@dask.delayed
def perform_write(df: pd.DataFrame, path: str):
    dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir, exist_ok=True)
    df.to_parquet(path)
    return len(df)


@dask.delayed
def get_partition_info_dict(write_results: dict[HealpixPixel, int]):
    return {pixel: (length, [pixel.pixel]) for pixel, length in write_results.items()}


def write_catalog(catalog, base_catalog_path: str, catalog_name: str = None):
    if not os.path.exists(base_catalog_path):
        os.makedirs(base_catalog_path)
    results = {}
    partitions = catalog._ddf.to_delayed()
    for pixel, partition_index in catalog._ddf_pixel_map.items():
        pixel_path = hipscat.io.paths.pixel_catalog_file(base_catalog_path, pixel.order, pixel.pixel)
        results[pixel] = perform_write(partitions[partition_index], pixel_path)

    partition_info_dict = get_partition_info_dict(results).compute()

    catalog_info = catalog.hc_structure.catalog_info
    if catalog_name is not None:
        catalog_info = dataclasses.replace(catalog_info, catalog_name=catalog_name)

    total_rows = sum(pi[0] for pi in partition_info_dict.values())
    catalog_info = dataclasses.replace(catalog_info, total_rows=total_rows)
    hipscat.io.write_catalog_info(base_catalog_path, catalog_info)
    hipscat.io.write_partition_info(base_catalog_path, partition_info_dict)