from lsdb import Catalog
from lsdb.catalog.catalog_source_type import CatalogSourceType
from lsdb.loaders.hipscat.catalog_loader_factory import \
    build_catalog_loader_for_source


def from_hipscat(name: str, source: CatalogSourceType, *args, **kwargs):
    """Load a catalog from a HiPSCat formatted catalog.

    Args:
        name: Name of the catalog
        source: Source to load the catalog from, currently supported options are:
            -`local`: HiPSCat files stored locally on disk
        local_path: For a local source, the path to find the root of the HiPSCat catalog
    """
    loader = build_catalog_loader_for_source(source, *args, **kwargs)
    ddf, ddf_pixel_map = loader.load_dask_df()
    hc_structure = loader.load_hipscat_catalog()
    source_info = loader.get_source_info_dict()
    return Catalog(name, ddf, ddf_pixel_map, hc_structure, source_info)
