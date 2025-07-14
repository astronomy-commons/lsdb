from pathlib import Path

import hats as hc
from hats.catalog.catalog_collection import CollectionProperties
from upath import UPath

from lsdb.io.to_hats import extra_property_dict


def to_collection(
    catalog,
    *,
    base_collection_path: str | Path | UPath,
    collection_name: str,
    default_columns: list[str] | None = None,
    overwrite: bool = False,
    **kwargs,
):
    """Saves the catalog collection to disk in the HATS format.

    The output contains the main catalog and its margin cache, if it exists.

    Args:
        base_catalog_path (str): Location where catalog is saved to
        collection_name (str): The name of the collection to be saved
        default_columns (list[str]): A metadata property with the list of the columns in the
            catalog to be loaded by default. By default, uses the default columns from the
            original hats catalog if they exist.
        overwrite (bool): If True existing collection is overwritten
        **kwargs: Arguments to pass to the parquet write operations
    """
    base_collection_path = hc.io.file_io.get_upath(base_collection_path) / collection_name
    catalog_name = catalog.name if catalog.name is not None else collection_name
    properties = {"obs_collection": collection_name, "hats_primary_table_url": catalog_name}

    catalog.to_hats(
        base_collection_path / catalog_name,
        catalog_name=catalog_name,
        default_columns=default_columns,
        overwrite=overwrite,
        **kwargs,
    )

    if catalog.margin is not None:
        margin_threshold = int(catalog.margin.hc_structure.catalog_info.margin_threshold)
        margin_name = (
            catalog.margin.name
            if catalog.margin.name is not None
            else f"{catalog_name}_{margin_threshold}arcs"
        )
        catalog.margin.to_hats(
            base_collection_path / margin_name,
            catalog_name=margin_name,
            default_columns=default_columns,
            overwrite=overwrite,
            **kwargs,
        )
        properties = properties | {"all_margins": margin_name, "default_margin": margin_name}

    properties = properties | extra_property_dict(base_collection_path)
    collection_info = CollectionProperties(**properties)
    collection_info.to_properties_file(base_collection_path)
