from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Type, Union

import hipscat as hc
from hipscat.catalog import CatalogType
from hipscat.catalog.dataset import BaseCatalogInfo
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hipscat.hipscat_loader_factory import get_loader_for_type
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig

dataset_class_for_catalog_type: Dict[CatalogType, Type[Dataset]] = {
    CatalogType.OBJECT: Catalog,
    CatalogType.SOURCE: Catalog,
    CatalogType.ASSOCIATION: AssociationCatalog,
    CatalogType.MARGIN: MarginCatalog,
}


# pylint: disable=unused-argument
def read_hipscat(
    path: str,
    catalog_type: Type[Dataset] | None = None,
    pixels_to_load: List[HealpixPixel] | None = None,
    storage_options: dict | None = None,
    columns: List[str] | None = None,
    margin_cache: MarginCatalog | None = None,
    **kwargs,
) -> Dataset:
    """Load a catalog from a HiPSCat formatted catalog.

    Typical usage example, where we load a catalog with a subset of columns:
        lsdb.read_hipscat(path="./my_catalog_dir", catalog_type=lsdb.Catalog, columns=["ra","dec"])

    Args:
        path (str): The path that locates the root of the HiPSCat catalog
        catalog_type (Type[Dataset]): Default `None`. By default, the type of the catalog is loaded
            from the catalog info and the corresponding object type is returned. Python's type hints
            cannot allow a return type specified by a loaded value, so to use the correct return
            type for type checking, the type of the catalog can be specified here. Use by specifying
            the lsdb class for that catalog.
        pixels_to_load (List[HealpixPixel]): The subset of catalog HEALPix to load for the main catalog
        storage_options (dict): Dictionary that contains abstract filesystem credentials
        columns (List[str]): Default `None`. The set of columns to filter the catalog on.
        margin_cache (MarginCatalog): The margin cache for the main catalog
        **kwargs: Arguments to pass to the pandas parquet file reader

    Returns:
        Catalog object loaded from the given parameters
    """

    # Creates a config object to store loading parameters from all keyword arguments.
    kwd_args = locals().copy()
    config_args = {field.name: kwd_args[field.name] for field in dataclasses.fields(HipscatLoadingConfig)}
    config = HipscatLoadingConfig(**config_args)

    catalog_type_to_use = _get_dataset_class_from_catalog_info(path, storage_options=storage_options)

    if catalog_type is not None:
        catalog_type_to_use = catalog_type

    loader = get_loader_for_type(catalog_type_to_use, path, config, storage_options=storage_options)
    return loader.load_catalog()


def read_hipscat_subset(
    path: str,
    catalog_type: Type[Dataset] | None = None,
    storage_options: dict | None = None,
    search_filter: AbstractSearch | None = None,
    n_files: int | None = None,
    order: int | None = None,
    **kwargs,
):
    """Load a catalog subset from a HiPSCat formatted catalog.

    Typical usage example, where we load a catalog from a cone search:
        lsdb.read_hipscat_subset(
            path="./my_catalog_dir",
            catalog_type=lsdb.Catalog,
            columns=["ra","dec"],
            filter=lsdb.core.search.ConeSearch(ra, dec, radius_arcsec),
        )

    Args:
        path (str): The path that locates the root of the HiPSCat catalog
        catalog_type (Type[Dataset]): Default `None`. By default, the type of the catalog is loaded
            from the catalog info and the corresponding object type is returned. Python's type hints
            cannot allow a return type specified by a loaded value, so to use the correct return
            type for type checking, the type of the catalog can be specified here. Use by specifying
            the lsdb class for that catalog.
        storage_options (dict): Dictionary that contains abstract filesystem credentials
        search_filter (Type[AbstractSearch]): The filter method to be applied to the catalog.
            By default, all files are considered.
        n_files (int): The number of files to read, in case no filter was specified.
            By default, all files are considered.
        order (int): The order of the files to read, in case n_files was specified.
            By default, the largest catalog order is used.
        **kwargs: Arguments to pass to the read hipscat call

    Returns:
        Catalog object loaded from the given parameters
    """
    pixels_to_load = None

    hc_structure = HCHealpixDataset.read_from_hipscat(path, storage_options)
    catalog_pixels = hc_structure.get_healpix_pixels()

    if isinstance(search_filter, AbstractSearch):
        pixels_to_load = search_filter.search_partitions(catalog_pixels)

    if n_files is not None:
        files_order = order if order is not None else hc_structure.partition_info.get_highest_order()
        pixels_of_order = [pixel for pixel in catalog_pixels if pixel.order == files_order]
        pixels_to_load = pixels_of_order[:n_files]

    return read_hipscat(
        path=path,
        catalog_type=catalog_type,
        pixels_to_load=pixels_to_load,
        storage_options=storage_options,
        **kwargs,
    )


def _get_dataset_class_from_catalog_info(
    base_catalog_path: str, storage_options: Union[Dict[Any, Any], None] = None
) -> Type[Dataset]:
    base_catalog_dir = hc.io.get_file_pointer_from_path(base_catalog_path)
    catalog_info_path = hc.io.paths.get_catalog_info_pointer(base_catalog_dir)
    catalog_info = BaseCatalogInfo.read_from_metadata_file(catalog_info_path, storage_options=storage_options)
    catalog_type = catalog_info.catalog_type
    if catalog_type not in dataset_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    return dataset_class_for_catalog_type[catalog_type]
