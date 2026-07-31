"""An ImageStore backed by image catalog rows and the format reader registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from lsdb.cutouts.image_store import ImageStore
from lsdb.cutouts.readers import get_image_reader, resolve_image_format

__all__ = ["CatalogImageStore"]


class CatalogImageStore(ImageStore):
    """Resolves image ids to pixels via image catalog rows and format readers.

    Built from (a partition of) an :class:`~lsdb.ImageCatalog`: each row
    provides the path, format and WCS of one image. Pixels are read lazily on
    first access through the reader registered for the image's format, and
    cached per store instance.

    Parameters
    ----------
    image_rows : pd.DataFrame
        Image catalog rows; must include ``image_id``, ``path`` and ``wcs``
        columns. Duplicate ids (an image appearing in several partitions)
        are deduplicated.
    default_format : str or None, default None
        The catalog-level ``image_format`` property, used when a row has no
        ``format`` value.
    """

    def __init__(self, image_rows: pd.DataFrame, default_format: str | None = None):
        rows = image_rows.drop_duplicates(subset="image_id")
        self._rows = {row["image_id"]: row for _, row in rows.iterrows()}
        self._default_format = default_format
        self._image_cache: dict[str, np.ndarray] = {}
        self._wcs_cache: dict[str, WCS] = {}

    def _row(self, image_id: str) -> pd.Series:
        if image_id not in self._rows:
            raise KeyError(f"Image '{image_id}' not found in this image store")
        return self._rows[image_id]

    def get_image(self, image_id: str) -> np.ndarray:
        if image_id not in self._image_cache:
            row = self._row(image_id)
            image_format = resolve_image_format(row.get("format"), self._default_format, row["path"])
            reader = get_image_reader(image_format)
            self._image_cache[image_id] = reader.read_image(row["path"])
        return self._image_cache[image_id]

    def get_wcs(self, image_id: str) -> WCS | None:
        # Imported here to avoid a circular import at module load time
        from lsdb.catalog.image_catalog import (  # pylint: disable=import-outside-toplevel
            wcs_from_params,
        )

        if image_id not in self._wcs_cache:
            self._wcs_cache[image_id] = wcs_from_params(self._row(image_id)["wcs"])
        return self._wcs_cache[image_id]

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._rows
