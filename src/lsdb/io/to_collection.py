from pathlib import Path

import hats as hc
from hats.catalog.catalog_collection import CollectionProperties
from upath import UPath

from lsdb.io.common import new_provenance_properties
from lsdb.io.to_hats import to_hats


def to_collection(
    catalog,
    *,
    base_collection_path: str | Path | UPath,
    catalog_name: str | None = None,
    default_columns: list[str] | None = None,
    overwrite: bool = False,
    error_if_empty: bool = True,
    **kwargs,
):
    """Saves the catalog collection to disk in the HATS format.

    The output contains the main catalog and its margin cache, if it exists.

    Parameters
    ----------
    catalog : HealpixDataset
        A catalog to export
    base_catalog_path : path-like
        Location where catalog is saved to
    catalog_name : str or None, default None
        The name of the catalog to be saved
    default_columns : list[str] or None, default None
        A metadata property with the list of the columns in the
        catalog to be loaded by default. By default, uses the default columns from the
        original hats catalog if they exist.
    overwrite : bool, default False
        If True existing collection is overwritten
    error_if_empty : bool, default True
        If True, raises an error if the catalog is empty
    **kwargs
        Arguments to pass to the parquet write operations
    """
    base_collection_path = hc.io.file_io.get_upath(base_collection_path)
    catalog_name = catalog_name if catalog_name else catalog.hc_structure.catalog_name
    properties = {"obs_collection": catalog_name, "hats_primary_table_url": catalog_name}

    to_hats(
        catalog,
        base_catalog_path=base_collection_path / catalog_name,
        catalog_name=catalog_name,
        default_columns=default_columns,
        overwrite=overwrite,
        error_if_empty=error_if_empty,
        **kwargs,
    )

    if catalog.margin is not None:
        margin_name = f"{catalog_name}_{int(catalog.margin.hc_structure.catalog_info.margin_threshold)}arcs"
        to_hats(
            catalog.margin,
            base_catalog_path=base_collection_path / margin_name,
            catalog_name=margin_name,
            default_columns=default_columns,
            overwrite=overwrite,
            error_if_empty=False,
            **kwargs,
        )
        properties = properties | {"all_margins": margin_name, "default_margin": margin_name}

    properties = properties | new_provenance_properties(base_collection_path)
    collection_info = CollectionProperties(**properties)
    collection_info.to_properties_file(base_collection_path)
