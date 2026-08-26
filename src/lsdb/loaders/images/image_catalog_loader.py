"""Loader that builds an ImageCatalog from per-image metadata.

Partitioning follows the HATS philosophy — no partition heavier than a
threshold — but with top-down refinement instead of the bottom-up histogram
aggregation used for point catalogs: image counts do not sum across sibling
pixels (an image overlapping all four children is one image in the parent,
not four), so pixels are recursively split while they overlap more than
``threshold`` images, bottoming out at ``highest_order``.

Footprints are computed as MOCs flattened to depth-29 HEALPix ranges — the
same interval representation used by the HATS pixel tree and by the
``_healpix_29`` spatial index — used in memory for partitioning and then
discarded: the catalog stores only the compact WCS parameters, from which
footprints are recomputed wherever needed.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import reduce

import hats as hc
import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from hats.catalog import CatalogType, TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, SPATIAL_INDEX_ORDER, compute_spatial_index
from mocpy import MOC

from lsdb.catalog.image_catalog import (
    ImageCatalog,
    image_footprint_moc,
    wcs_from_params,
    wcs_to_params,
)
from lsdb.io.common import new_provenance_properties
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.dataframe.from_dataframe_utils import _generate_op

__all__ = ["ImageCatalogLoader"]


def _ranges_intersect_interval(ranges: np.ndarray, start: int, end: int) -> bool:
    """Whether any of the sorted, disjoint [start, end) `ranges` overlaps [start, end)."""
    below = np.searchsorted(ranges[:, 1], start, side="right")
    above = np.searchsorted(ranges[:, 0], end, side="left")
    return bool(above > below)


class ImageCatalogLoader:
    """Builds an ImageCatalog from a dataframe of image metadata.

    Parameters
    ----------
    images : pd.DataFrame
        One row per image with at least the columns ``image_id``, ``path``,
        ``width`` and ``height``. Any extra columns (band, mjd, format, ...)
        are kept. If ``wcs`` is not given, a ``wcs`` column of FITS header
        strings must be present.
    wcs : sequence of astropy.wcs.WCS, optional
        The WCS of each image, aligned with the rows of ``images``.
    threshold : int, default 100
        Maximum number of images per partition. Pixels overlapping more
        images are split until the count drops below the threshold or
        ``highest_order`` is reached.
    lowest_order : int, default 0
        The coarsest HEALPix order partitions may have.
    highest_order : int, default 8
        The finest HEALPix order partitions may have. Must not exceed
        ``moc_order``.
    moc_order : int, default 11
        HEALPix order at which image footprint MOCs are computed for
        partitioning. Recorded in the catalog properties as the suggested
        order for match-time footprints.
    catalog_name : str, default "image_catalog"
        Name for the catalog.
    image_format : str or None, default None
        Default storage format of the images (e.g. ``fits``, ``zarr``),
        recorded in the catalog properties. Individual images may override
        it with a ``format`` column.
    """

    def __init__(
        self,
        images: pd.DataFrame,
        wcs: Sequence[WCS] | None = None,
        *,
        threshold: int = 100,
        lowest_order: int = 0,
        highest_order: int = 8,
        moc_order: int = 11,
        catalog_name: str = "image_catalog",
        image_format: str | None = None,
    ):
        if highest_order > moc_order:
            raise ValueError(f"highest_order ({highest_order}) must not exceed moc_order ({moc_order})")
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.images = images.reset_index(drop=True).copy()
        self.wcs_objects = self._resolve_wcs(wcs)
        self.threshold = threshold
        self.lowest_order = lowest_order
        self.highest_order = highest_order
        self.moc_order = moc_order
        self.catalog_name = catalog_name
        self.image_format = image_format

    def _resolve_wcs(self, wcs: Sequence[WCS] | None) -> list[WCS]:
        if wcs is not None:
            if len(wcs) != len(self.images):
                raise ValueError(f"Got {len(wcs)} WCS objects for {len(self.images)} images")
            wcs_objects = list(wcs)
        elif "wcs" in self.images.columns:
            wcs_objects = [wcs_from_params(value) for value in self.images["wcs"]]
        else:
            raise ValueError(
                "Provide `wcs` objects or a `wcs` column of parameter dicts / FITS header strings"
            )
        # Normalize to the compact parameter representation stored on disk
        self.images["wcs"] = pd.Series([wcs_to_params(w) for w in wcs_objects], index=self.images.index)
        return wcs_objects

    def load_catalog(self) -> ImageCatalog:
        """Build the ImageCatalog.

        Returns
        -------
        ImageCatalog
        """
        footprint_mocs = self._prepare_image_columns()
        footprint_ranges = [np.asarray(moc.to_depth29_ranges, dtype=np.int64) for moc in footprint_mocs]
        self._validate_columns()

        pixel_map = self._partition_images(footprint_ranges)
        pixels = list(pixel_map.keys())
        partitions = [self._build_partition(rows) for rows in pixel_map.values()]
        pixel_order = get_pixel_argsort(pixels)
        ordered_pixels = [pixels[i] for i in pixel_order]
        ordered_partitions = [partitions[i] for i in pixel_order]

        op, total_rows = _generate_op(ordered_partitions, ordered_pixels)
        catalog_info = self._create_catalog_info(total_rows)
        coverage = reduce(lambda a, b: a.union(b), footprint_mocs) if footprint_mocs else None
        hc_structure = hc.catalog.Catalog(
            catalog_info,
            ordered_pixels,
            moc=coverage,
            schema=get_arrow_schema(op.meta),
            generate_snapshot=True,
        )
        return ImageCatalog(op, hc_structure, loading_config=None)

    def _prepare_image_columns(self) -> list[MOC]:
        """Add the ra/dec (image center) columns; return the footprint MOCs."""
        centers = [
            w.pixel_to_world((naxis1 - 1) / 2, (naxis2 - 1) / 2)
            for w, naxis1, naxis2 in zip(self.wcs_objects, self.images["width"], self.images["height"])
        ]
        self.images["ra"] = [center.ra.deg for center in centers]
        self.images["dec"] = [center.dec.deg for center in centers]
        return [
            image_footprint_moc(w, naxis1, naxis2, self.moc_order)
            for w, naxis1, naxis2 in zip(self.wcs_objects, self.images["width"], self.images["height"])
        ]

    def _validate_columns(self):
        # Imported here to avoid a circular import at module load time
        from lsdb.catalog.image_catalog import (  # pylint: disable=import-outside-toplevel
            REQUIRED_IMAGE_COLUMNS,
        )

        missing = [column for column in REQUIRED_IMAGE_COLUMNS if column not in self.images.columns]
        if missing:
            raise ValueError(f"Image dataframe is missing required columns: {missing}")

    def _partition_images(self, footprint_ranges: list[np.ndarray]) -> dict[HealpixPixel, list[int]]:
        """Assign images to partitions by top-down threshold refinement.

        Returns
        -------
        dict[HealpixPixel, list[int]]
            Mapping of partition pixel to the positions of the images
            overlapping it.
        """
        all_images = list(range(len(footprint_ranges)))
        pixel_map: dict[HealpixPixel, list[int]] = {}
        # Seed with the lowest-order pixels; children only ever test their parent's images
        stack = [
            (HealpixPixel(self.lowest_order, pixel_index), all_images)
            for pixel_index in range(12 * 4**self.lowest_order)
        ]
        while stack:
            pixel, candidates = stack.pop()
            shift = 2 * (SPATIAL_INDEX_ORDER - pixel.order)
            start, end = pixel.pixel << shift, (pixel.pixel + 1) << shift
            overlapping = [
                position
                for position in candidates
                if _ranges_intersect_interval(footprint_ranges[position], start, end)
            ]
            if not overlapping:
                continue
            if len(overlapping) > self.threshold and pixel.order < self.highest_order:
                stack.extend(
                    (HealpixPixel(pixel.order + 1, 4 * pixel.pixel + child), overlapping)
                    for child in range(4)
                )
                continue
            if len(overlapping) > self.threshold:
                warnings.warn(
                    f"Partition {pixel} overlaps {len(overlapping)} images, more than the "
                    f"threshold of {self.threshold}; increase highest_order to split further."
                )
            pixel_map[pixel] = overlapping
        return pixel_map

    def _build_partition(self, row_positions: list[int]) -> npd.NestedFrame:
        """Build one partition: the image rows indexed by the spatial index of their centers."""
        partition = self.images.iloc[row_positions].copy()
        partition[SPATIAL_INDEX_COLUMN] = compute_spatial_index(
            ra_values=partition["ra"].to_list(), dec_values=partition["dec"].to_list()
        )
        return npd.NestedFrame(partition.set_index(SPATIAL_INDEX_COLUMN).sort_index())

    def _create_catalog_info(self, total_rows: int) -> TableProperties:
        return TableProperties(
            catalog_name=self.catalog_name,
            catalog_type=CatalogType.IMAGE,
            ra_column="ra",
            dec_column="dec",
            healpix_column=SPATIAL_INDEX_COLUMN,
            healpix_order=SPATIAL_INDEX_ORDER,
            total_rows=total_rows,
            image_format=self.image_format,
            image_moc_order=self.moc_order,
            **new_provenance_properties(),
        )
