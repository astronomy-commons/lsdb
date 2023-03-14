from lsdb.catalog.catalog_source_type import CatalogSourceType


# pylint: disable=unused-argument
def infer_source_from_path(path: str) -> CatalogSourceType:
    """Infer the type of source from the path string

    Currently supports these sources:
        - local files

    Args:
        path - Path string to infer source from

    Returns:
        `CatalogSourceType` corresponding to inferred source
    """
    return CatalogSourceType.LOCAL
