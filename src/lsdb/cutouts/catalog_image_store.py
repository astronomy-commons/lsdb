"""An ImageStore backed by image catalog rows and the format reader registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from lsdb.cutouts.image_store import ImageStore
from lsdb.cutouts.readers import get_image_reader, resolve_image_format

__all__ = ["CatalogImageStore"]

READ_MODES = ("auto", "full", "region")


class CatalogImageStore(ImageStore):
    """Resolves image ids to pixels via image catalog rows and format readers.

    Built from (a partition of) an :class:`~lsdb.ImageCatalog`: each row
    provides the path, format and WCS of one image. Pixels are read lazily on
    first access through the reader registered for the image's format.

    Region reads (``read_mode``) control how much of a file is read when a
    single cutout is rendered:

    - ``"auto"`` (default): regions are read individually (for tiled formats,
      only the intersecting tiles/chunks) until an image has served
      ``full_read_threshold`` region requests, after which the full image is
      loaded once and cached — cheap for a few stamps, amortized for many.
    - ``"region"``: always read regions individually; never load full images.
    - ``"full"``: always load and cache full images (regions are views).

    Parameters
    ----------
    image_rows : pd.DataFrame
        Image catalog rows; must include ``image_id``, ``path`` and ``wcs``
        columns. Duplicate ids (an image appearing in several partitions)
        are deduplicated.
    default_format : str or None, default None
        The catalog-level ``image_format`` property, used when a row has no
        ``format`` value.
    read_mode : str, default "auto"
        One of ``auto``, ``full``, ``region`` (see above).
    full_read_threshold : int, default 8
        In ``auto`` mode, the number of region requests after which an
        image is loaded in full.
    """

    def __init__(
        self,
        image_rows: pd.DataFrame,
        default_format: str | None = None,
        read_mode: str = "auto",
        full_read_threshold: int = 8,
    ):
        if read_mode not in READ_MODES:
            raise ValueError(f"read_mode must be one of {READ_MODES}, got '{read_mode}'")
        rows = image_rows.drop_duplicates(subset="image_id")
        self._rows = {row["image_id"]: row for _, row in rows.iterrows()}
        self._default_format = default_format
        self._read_mode = read_mode
        self._full_read_threshold = full_read_threshold
        self._image_cache: dict[str, np.ndarray] = {}
        self._wcs_cache: dict[str, WCS] = {}
        self._region_cache: dict[tuple, np.ndarray] = {}
        self._region_counts: dict[str, int] = {}

    def _row(self, image_id: str) -> pd.Series:
        if image_id not in self._rows:
            raise KeyError(f"Image '{image_id}' not found in this image store")
        return self._rows[image_id]

    def _reader_for(self, image_id: str):
        row = self._row(image_id)
        image_format = resolve_image_format(row.get("format"), self._default_format, row["path"])
        return get_image_reader(image_format), row["path"]

    def get_image(self, image_id: str) -> np.ndarray:
        if image_id not in self._image_cache:
            reader, path = self._reader_for(image_id)
            self._image_cache[image_id] = reader.read_image(path)
        return self._image_cache[image_id]

    def get_region(self, image_id: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        # A fully-cached image always wins: regions are zero-copy views into it
        if image_id in self._image_cache or self._read_mode == "full":
            image = self.get_image(image_id)
            return image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        requests = self._region_counts[image_id] = self._region_counts.get(image_id, 0) + 1
        if self._read_mode == "auto" and requests > self._full_read_threshold:
            # This image is hot; one full read now amortizes all later regions
            image = self.get_image(image_id)
            return image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        key = (image_id, y0, y1, x0, x1)
        if key not in self._region_cache:
            reader, path = self._reader_for(image_id)
            self._region_cache[key] = reader.read_region(path, y0, y1, x0, x1)
        return self._region_cache[key]

    def plan_reads(self, image_ids) -> None:
        """Load images in full upfront when the announced batch makes them hot.

        In ``auto`` mode, any image that will serve more than
        ``full_read_threshold`` cutouts is loaded in full immediately, so the
        batch never pays for region reads that a full load would supersede.
        No-op in ``region`` mode (never load full) and effectively free in
        ``full`` mode (the first region request loads the image anyway).

        Parameters
        ----------
        image_ids : iterable of str
            The image id of every cutout about to be rendered.
        """
        if self._read_mode != "auto":
            return
        counts: dict[str, int] = {}
        for image_id in image_ids:
            counts[image_id] = counts.get(image_id, 0) + 1
        for image_id, count in counts.items():
            if count > self._full_read_threshold and image_id not in self._image_cache:
                self.get_image(image_id)

    def cache_info(self, human: bool = False) -> dict:
        """Memory accounting for the pixel caches.

        Parameters
        ----------
        human : bool, default False
            If True, byte values are formatted as human-readable strings
            (e.g. ``"92.5 MB"``) instead of integers.

        Returns
        -------
        dict
            ``full_images``/``full_bytes``: count and bytes of fully-loaded
            cached images; ``regions``/``region_bytes``: count and bytes of
            cached region reads; ``total_bytes``: their sum.
        """
        from human_readable import file_size  # pylint: disable=import-outside-toplevel

        full_bytes = sum(image.nbytes for image in self._image_cache.values())
        region_bytes = sum(region.nbytes for region in self._region_cache.values())

        def size(value: int) -> int | str:
            return file_size(value) if human else value

        return {
            "full_images": len(self._image_cache),
            "full_bytes": size(full_bytes),
            "regions": len(self._region_cache),
            "region_bytes": size(region_bytes),
            "total_bytes": size(full_bytes + region_bytes),
        }

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
