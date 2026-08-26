"""Store-backed image cutout support for LSDB catalogs.

The pandas-level classes (:class:`CutoutArray`, :class:`CutoutSeries`, the
image store interface) live in nested-pandas; this package adds the
astronomy-facing pieces: WCS-aware stores, image format readers, the
:class:`CatalogImageStore` over image catalog rows, and the per-partition
object-to-image matching used by ``Catalog.add_cutouts``.
"""

from nested_pandas import CutoutArray, CutoutDtype, CutoutSeries
from nested_pandas.tensor.cutouts import CUTOUT_DESCRIPTOR_TYPE

from .catalog_image_store import CatalogImageStore
from .coverage_map import CoverageMap
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
    "CutoutSeries",
    "CUTOUT_DESCRIPTOR_TYPE",
    "ImageStore",
    "InMemoryImageStore",
    "ChainImageStore",
    "merge_stores",
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
