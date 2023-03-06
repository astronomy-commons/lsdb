from enum import Enum


class CatalogSourceType(Enum):
    """Options for the source to load a HiPSCat Catalog from"""
    LOCAL = 'local'
    S3 = 's3'
    HTTP = 'http'
