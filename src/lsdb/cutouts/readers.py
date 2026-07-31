"""Format-specific image readers and the reader registry.

An :class:`ImageReader` turns a path into a pixel array. Readers are
registered per format string (``fits``, ``zarr``, ...) so image catalogs can
reference pixel data in any storage format; the format of each image resolves
in order of precedence: per-row ``format`` column, catalog-level
``image_format`` property, then the file extension.

WCS never comes from the files — image catalogs carry it as a column — so
readers only deliver pixels, which is what makes formats without a WCS
convention (like Zarr) first-class citizens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from upath import UPath

__all__ = [
    "ImageReader",
    "FitsImageReader",
    "ZarrImageReader",
    "register_image_reader",
    "get_image_reader",
    "resolve_image_format",
]

_EXTENSION_TO_FORMAT = {
    ".fits": "fits",
    ".fit": "fits",
    ".fts": "fits",
    ".fz": "fits",
    ".zarr": "zarr",
}


class ImageReader(ABC):
    """Reads pixel arrays from image files of one storage format."""

    @abstractmethod
    def read_image(self, path: str) -> np.ndarray:
        """Read the full pixel array of an image.

        Parameters
        ----------
        path : str
            Path or URI of the image file.

        Returns
        -------
        np.ndarray
            2D pixel array.
        """


class FitsImageReader(ImageReader):
    """Reads FITS images with astropy, using fsspec for remote paths.

    Parameters
    ----------
    hdu_index : int or None, default None
        Index of the HDU holding the pixels. If None, the first HDU with
        image data is used.
    """

    def __init__(self, hdu_index: int | None = None):
        self.hdu_index = hdu_index

    def read_image(self, path: str) -> np.ndarray:
        from astropy.io import fits  # pylint: disable=import-outside-toplevel

        path = str(path)
        if path.startswith("file://"):
            # Open local files directly; astropy would otherwise treat the URL
            # through its download-to-cache machinery (copying the whole file)
            path = path.removeprefix("file://")
        use_fsspec = "://" in path
        with fits.open(path, use_fsspec=use_fsspec) as hdu_list:
            if self.hdu_index is not None:
                return np.asarray(hdu_list[self.hdu_index].data)
            for hdu in hdu_list:
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    return np.asarray(hdu.data)
        raise ValueError(f"No 2D image HDU found in {path}")


class ZarrImageReader(ImageReader):
    """Reads Zarr images (requires the optional ``zarr`` dependency).

    Parameters
    ----------
    array_key : str, default "image"
        Name of the array to read when the path points to a Zarr group.
        Ignored when the path points directly to an array.
    """

    def __init__(self, array_key: str = "image"):
        self.array_key = array_key

    def read_image(self, path: str) -> np.ndarray:
        try:
            import zarr  # pylint: disable=import-outside-toplevel
        except ImportError as exception:
            raise ImportError(
                "Reading zarr images requires the 'zarr' package: pip install zarr"
            ) from exception
        node = zarr.open(path, mode="r")
        if hasattr(node, "keys"):  # a group; read the named array
            node = node[self.array_key]
        return np.asarray(node[:])


_READERS: dict[str, ImageReader] = {
    "fits": FitsImageReader(),
    "zarr": ZarrImageReader(),
}


def register_image_reader(image_format: str, reader: ImageReader) -> None:
    """Register (or replace) the reader for a storage format.

    Parameters
    ----------
    image_format : str
        Format name as used in image catalogs (e.g. ``fits``, ``zarr``,
        ``hdf5``).
    reader : ImageReader
        The reader instance handling that format.
    """
    _READERS[image_format] = reader


def get_image_reader(image_format: str) -> ImageReader:
    """Return the registered reader for a storage format.

    Parameters
    ----------
    image_format : str
        The format name.

    Returns
    -------
    ImageReader

    Raises
    ------
    ValueError
        If no reader is registered for the format.
    """
    if image_format not in _READERS:
        raise ValueError(
            f"No image reader registered for format '{image_format}'; "
            f"available formats: {sorted(_READERS)}. Use register_image_reader to add one."
        )
    return _READERS[image_format]


def resolve_image_format(row_format: str | None, catalog_format: str | None, path: str) -> str:
    """Resolve the storage format of one image.

    Precedence: the image row's ``format`` value, then the catalog-level
    ``image_format`` property, then the file extension.

    Parameters
    ----------
    row_format : str or None
        Value of the image row's ``format`` column, if any.
    catalog_format : str or None
        The catalog's ``image_format`` property, if any.
    path : str
        The image path, used for extension sniffing as a last resort.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    if row_format:
        return row_format
    if catalog_format:
        return catalog_format
    suffix = UPath(path).suffix.lower()
    if suffix in _EXTENSION_TO_FORMAT:
        return _EXTENSION_TO_FORMAT[suffix]
    raise ValueError(
        f"Cannot determine image format of '{path}': no format column, no catalog "
        "image_format property, and unrecognized extension."
    )
