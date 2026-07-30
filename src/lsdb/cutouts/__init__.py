"""Prototype support for image cutout columns in LSDB catalogs.

Cutout columns store lightweight descriptors (image id + pixel bounding box)
per row, plus one shared image store per array that resolves ids to pixels at
render time. See ``CutoutArray`` for the storage model.
"""

from .cutout_array import CUTOUT_ARROW_TYPE, CutoutArray, CutoutDtype, CutoutRef
from .cutout_series import CutoutAccessor, CutoutSeries
from .display import nestedframe_html, register_ipython_formatter, series_html, series_repr
from .image_store import ChainImageStore, ImageStore, InMemoryImageStore, merge_stores

# Make NestedFrame render cutout thumbnails in notebooks (no-op outside IPython).
register_ipython_formatter()

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
    "nestedframe_html",
    "register_ipython_formatter",
    "series_html",
    "series_repr",
]
