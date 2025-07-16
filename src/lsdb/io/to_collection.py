from pathlib import Path

import hats as hc
from hats.catalog.catalog_collection import CollectionProperties
from upath import UPath

from lsdb.catalog.dataset.dataset import Dataset


def to_collection(
    catalog,
    *,
    base_collection_path: str | Path | UPath,
    catalog_name: str | None = None,
    default_columns: list[str] | None = None,
    overwrite: bool = False,
    **kwargs,
):
    """Saves the catalog collection to disk in the HATS format.

    The output contains the main catalog and its margin cache, if it exists.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_path (str): Location where catalog is saved to
        catalog_name (str): The name of the catalog to be saved
        default_columns (list[str]): A metadata property with the list of the columns in the
            catalog to be loaded by default. By default, uses the default columns from the
            original hats catalog if they exist.
        overwrite (bool): If True existing collection is overwritten
        **kwargs: Arguments to pass to the parquet write operations
    """
    base_collection_path = hc.io.file_io.get_upath(base_collection_path)
    catalog_name = catalog_name if catalog_name else catalog.hc_structure.catalog_name
    properties = {"obs_collection": catalog_name, "hats_primary_table_url": catalog_name}

    catalog.to_hats(
        base_collection_path / catalog_name,
        catalog_name=catalog_name,
        default_columns=default_columns,
        overwrite=overwrite,
        as_collection=False,
        **kwargs,
    )

    if catalog.margin is not None and len(catalog.margin.get_healpix_pixels()) > 0:
        margin_name = f"{catalog_name}_{int(catalog.margin.hc_structure.catalog_info.margin_threshold)}arcs"
        catalog.margin.to_hats(
            base_collection_path / margin_name,
            catalog_name=margin_name,
            default_columns=default_columns,
            overwrite=overwrite,
            **kwargs,
        )
        properties = properties | {"all_margins": margin_name, "default_margin": margin_name}

    properties = properties | Dataset.new_provenance_properties(base_collection_path)
    collection_info = CollectionProperties(**properties)
    collection_info.to_properties_file(base_collection_path)
