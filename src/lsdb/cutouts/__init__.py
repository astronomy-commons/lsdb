"""Prototype support for image cutout columns in LSDB catalogs.

Cutout columns store lightweight descriptors (image id + pixel bounding box)
per row, plus one shared image store per array that resolves ids to pixels at
render time. See ``CutoutArray`` for the storage model.
"""

from .catalog_image_store import CatalogImageStore
from .coverage_map import CoverageMap
from .cutout_array import CUTOUT_ARROW_TYPE, CutoutArray, CutoutDtype, CutoutRef
from .cutout_series import CutoutAccessor, CutoutSeries
from .display import cutout_cell_html, series_html, series_repr
from .image_store import ChainImageStore, ImageStore, InMemoryImageStore, merge_stores
from .matching import match_partition
from .readers import (
    FitsImageReader,
    ImageReader,
    ZarrImageReader,
    get_image_reader,
    register_image_reader,
    resolve_image_format,
)

__all__ = [
    "CutoutArray",
    "CutoutDtype",
    "CutoutRef",
    "CutoutSeries",
    "CutoutAccessor",
    "CUTOUT_ARROW_TYPE",
    "ImageStore",
    "InMemoryImageStore",
    "ChainImageStore",
    "merge_stores",
    "cutout_cell_html",
    "series_html",
    "series_repr",
    "CatalogImageStore",
    "ImageReader",
    "FitsImageReader",
    "ZarrImageReader",
    "register_image_reader",
    "get_image_reader",
    "resolve_image_format",
    "CoverageMap",
    "match_partition",
]
