from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from astropy.wcs import WCS

from lsdb.catalog.image_catalog import ImageCatalog
from lsdb.loaders.images.image_catalog_loader import ImageCatalogLoader

__all__ = ["from_images"]


def from_images(
    images: pd.DataFrame,
    wcs: Sequence[WCS] | None = None,
    *,
    threshold: int = 100,
    lowest_order: int = 0,
    highest_order: int = 8,
    moc_order: int = 11,
    catalog_name: str = "image_catalog",
    image_format: str | None = None,
) -> ImageCatalog:
    """Build an ImageCatalog from per-image metadata.

    Every image is assigned to all partitions its footprint overlaps, with
    partitions refined until none overlaps more than ``threshold`` images
    (or ``highest_order`` is reached). This is intended for prototyping and
    moderate image counts; large-scale catalog generation belongs in a
    hats-import pipeline.

    Parameters
    ----------
    images : pd.DataFrame
        One row per image with at least the columns ``image_id``, ``path``,
        ``width`` and ``height``. Any extra columns (band, mjd, format, ...)
        are kept. If ``wcs`` is not given, a ``wcs`` column of FITS header
        strings must be present.
    wcs : sequence of astropy.wcs.WCS, optional
        The WCS of each image, aligned with the rows of ``images``. Stored as
        compact WCS parameters in the ``wcs`` column.
    threshold : int, default 100
        Maximum number of images per partition.
    lowest_order : int, default 0
        The coarsest HEALPix order partitions may have.
    highest_order : int, default 8
        The finest HEALPix order partitions may have.
    moc_order : int, default 11
        HEALPix order at which image footprint MOCs are computed for
        partitioning.
    catalog_name : str, default "image_catalog"
        Name for the catalog.
    image_format : str or None, default None
        Default storage format of the images (e.g. ``fits``, ``zarr``).
        Individual images may override it with a ``format`` column.

    Returns
    -------
    ImageCatalog

    Examples
    --------
    >>> import lsdb
    >>> import pandas as pd
    >>> from astropy.wcs import WCS
    >>> wcs = WCS(naxis=2)
    >>> wcs.wcs.crpix = [50, 50]
    >>> wcs.wcs.cdelt = [-0.005, 0.005]
    >>> wcs.wcs.crval = [180.0, -30.0]
    >>> wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    >>> images = pd.DataFrame(
    ...     {"image_id": ["v1"], "path": ["s3://survey/v1.fits"], "width": [100], "height": [100]}
    ... )
    >>> catalog = lsdb.from_images(images, [wcs], catalog_name="demo")
    >>> catalog.hc_structure.catalog_info.catalog_type
    'image'
    """
    loader = ImageCatalogLoader(
        images,
        wcs,
        threshold=threshold,
        lowest_order=lowest_order,
        highest_order=highest_order,
        moc_order=moc_order,
        catalog_name=catalog_name,
        image_format=image_format,
    )
    return loader.load_catalog()
