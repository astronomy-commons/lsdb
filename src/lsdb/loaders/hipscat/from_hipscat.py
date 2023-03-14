import dataclasses

from lsdb import Catalog
from lsdb.catalog.catalog_source_type import CatalogSourceType
from lsdb.loaders.hipscat.hipscat_catalog_loader_factory import \
    build_hipscat_catalog_loader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig
from lsdb.loaders.path_utils import infer_source_from_path


def from_hipscat(
    path: str,
    source: CatalogSourceType = None,
) -> Catalog:
    """Load a catalog from a HiPSCat formatted catalog.

    Args:
        path: The path that locates the root of the HiPSCat catalog
        source: Source to load the catalog from. Default is `None`, in which case the source is
        inferred from the path. Currently supported options are:
            -`'local'`: HiPSCat files stored locally on disk
    """

    # Creates a config object to store loading parameters from all keyword arguments. I
    # originally had a few parameters in here, but after changing the file loading implementation
    # they weren't needed, so this object is now empty. But I wanted to keep this here for future
    # use
    kwd_args = locals().copy()
    config_args = {
        field.name: kwd_args[field.name]
        for field in dataclasses.fields(HipscatLoadingConfig)
    }
    config = HipscatLoadingConfig(**config_args)

    if source is None:
        source = infer_source_from_path(path)

    loader = build_hipscat_catalog_loader(path, source, config)

    return loader.load_catalog()
