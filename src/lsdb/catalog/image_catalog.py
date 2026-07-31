"""ImageCatalog: a HATS-partitioned catalog of survey image metadata.

Each row describes one image: an id, a path to the pixel data (any registered
format), its dimensions and its WCS. A row is written to *every* partition
whose HEALPix pixel overlaps the image footprint, so any partition locally
knows all images that could serve cutouts for its objects — rows are
duplicated across partitions by design.

The WCS is stored as a compact struct of parameters (``wcs`` column): CTYPE,
CRVAL, CRPIX and the CD matrix, roughly 90 bytes per row, with an optional
``extra`` field carrying a full FITS header string only for WCS with
distortion terms (SIP, lookup tables). Storing WCS in the catalog — rather
than reading it from the image files — is what makes image formats without a
WCS convention (like Zarr) first-class. Footprints are *not* stored; they are
recomputed from the WCS wherever needed.

Image catalogs are built with :func:`lsdb.from_images` and loaded through the
regular ``lsdb.open_catalog`` / ``lsdb.read_hats`` entry points (they carry the
HATS ``image`` catalog type).
"""

from __future__ import annotations

from collections.abc import Mapping

import hats as hc
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from mocpy import MOC

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset

# Columns every image catalog must provide.
REQUIRED_IMAGE_COLUMNS = ["image_id", "path", "width", "height", "wcs", "ra", "dec"]


def wcs_to_header_string(wcs: WCS) -> str:
    """Serialize a WCS to a FITS header string (round-trips distortion keywords).

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS to serialize.

    Returns
    -------
    str
        FITS header cards as a single string.
    """
    return wcs.to_header_string(relax=True)


def wcs_from_header_string(header_string: str) -> WCS:
    """Reconstruct a WCS from a FITS header string.

    Parameters
    ----------
    header_string : str
        FITS header cards as produced by :func:`wcs_to_header_string`.

    Returns
    -------
    astropy.wcs.WCS
    """
    return WCS(fits.Header.fromstring(header_string))


def wcs_to_params(wcs: WCS) -> dict:
    """Serialize a WCS to the compact parameter dict stored in image catalogs.

    Plain celestial WCS reduce to CTYPE, CRVAL, CRPIX and the CD matrix
    (``extra`` is None). WCS with distortion terms additionally carry the
    full FITS header string in ``extra``, which takes precedence when
    deserializing so no fidelity is lost.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS to serialize.

    Returns
    -------
    dict
        Keys ``ctype1``, ``ctype2``, ``crval1``, ``crval2``, ``crpix1``,
        ``crpix2``, ``cd1_1``, ``cd1_2``, ``cd2_1``, ``cd2_2``, ``extra``.
    """
    cd_matrix = wcs.pixel_scale_matrix  # normalizes CD vs CDELT+PC conventions
    radesys = getattr(wcs.wcs, "radesys", "") or ""
    needs_extra = wcs.has_distortion or radesys.upper() not in ("", "ICRS")
    return {
        "ctype1": wcs.wcs.ctype[0],
        "ctype2": wcs.wcs.ctype[1],
        "crval1": float(wcs.wcs.crval[0]),
        "crval2": float(wcs.wcs.crval[1]),
        "crpix1": float(wcs.wcs.crpix[0]),
        "crpix2": float(wcs.wcs.crpix[1]),
        "cd1_1": float(cd_matrix[0, 0]),
        "cd1_2": float(cd_matrix[0, 1]),
        "cd2_1": float(cd_matrix[1, 0]),
        "cd2_2": float(cd_matrix[1, 1]),
        "extra": wcs_to_header_string(wcs) if needs_extra else None,
    }


def wcs_from_params(value: Mapping | str) -> WCS:
    """Reconstruct a WCS from its stored representation.

    Parameters
    ----------
    value : Mapping or str
        A parameter dict as produced by :func:`wcs_to_params` (the ``extra``
        header string takes precedence when present), or a plain FITS header
        string.

    Returns
    -------
    astropy.wcs.WCS
    """
    if isinstance(value, str):
        return wcs_from_header_string(value)
    if value.get("extra"):
        return wcs_from_header_string(value["extra"])
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [value["ctype1"], value["ctype2"]]
    wcs.wcs.crval = [value["crval1"], value["crval2"]]
    wcs.wcs.crpix = [value["crpix1"], value["crpix2"]]
    wcs.wcs.cd = [[value["cd1_1"], value["cd1_2"]], [value["cd2_1"], value["cd2_2"]]]
    return wcs


def image_footprint_moc(wcs: WCS, width: int, height: int, order: int) -> MOC:
    """Compute the sky coverage of an image as a MOC at the given order.

    The footprint is the spherical polygon through the four image corners
    (pixel edges, not centers).

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The image WCS.
    width : int
        Image width in pixels (NAXIS1).
    height : int
        Image height in pixels (NAXIS2).
    order : int
        HEALPix order of the returned MOC.

    Returns
    -------
    mocpy.MOC
    """
    corners = wcs.calc_footprint(axes=(width, height), center=False)
    return MOC.from_polygon_skycoord(SkyCoord(corners, unit="deg"), max_depth=order)


class ImageCatalog(HealpixDataset):
    """A HATS catalog of image metadata, partitioned by footprint overlap.

    Every partition contains one row per image whose footprint overlaps that
    partition's HEALPix pixel; images spanning multiple pixels appear in each
    of them. The ``ra``/``dec`` columns hold the image *center* (used for the
    spatial index) and ``wcs`` holds the compact WCS parameter struct.

    Note that point-source spatial search semantics do not apply to extended
    footprints, which is why this class does not expose them.
    """

    hc_structure: hc.catalog.Catalog

    def wcs_for(self, row: pd.Series | dict) -> WCS:
        """Reconstruct the WCS of an image row.

        Parameters
        ----------
        row : pd.Series or dict
            A row of this catalog (must include the ``wcs`` column).

        Returns
        -------
        astropy.wcs.WCS
        """
        return wcs_from_params(row["wcs"])
