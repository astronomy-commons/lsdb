import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from hats.catalog import CatalogType
from hats.pixel_math.spatial_index import SPATIAL_INDEX_ORDER

import lsdb
from lsdb import ImageCatalog, from_images
from lsdb.catalog.image_catalog import (
    image_footprint_moc,
    wcs_from_header_string,
    wcs_from_params,
    wcs_to_header_string,
    wcs_to_params,
)

MOC_ORDER = 11


def make_wcs(ra, dec, pixscale_deg=0.005, size=100):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2, size / 2]
    wcs.wcs.cdelt = [-pixscale_deg, pixscale_deg]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def images_df():
    return pd.DataFrame(
        {
            "image_id": ["v1", "v2", "v3"],
            "path": ["s3://fake/v1.fits", "s3://fake/v2.fits", "s3://fake/v3.fits"],
            "width": [100, 100, 100],
            "height": [100, 100, 100],
            "band": ["g", "r", "g"],
        }
    )


@pytest.fixture
def wcs_list():
    # v1 and v2 overlap on the sky; v3 is far away
    return [make_wcs(180.0, -30.0), make_wcs(180.3, -30.0), make_wcs(185.0, -32.0)]


@pytest.fixture
def image_catalog(images_df, wcs_list):
    return from_images(
        images_df, wcs_list, moc_order=MOC_ORDER, catalog_name="demo_images", image_format="fits"
    )


def test_wcs_string_round_trip():
    wcs = make_wcs(180.0, -30.0)
    restored = wcs_from_header_string(wcs_to_header_string(wcs))
    assert wcs.pixel_to_world(10, 20).separation(restored.pixel_to_world(10, 20)).arcsec < 1e-9


def test_footprint_moc_contains_image():
    wcs = make_wcs(180.0, -30.0)
    moc = image_footprint_moc(wcs, 100, 100, MOC_ORDER)
    center = wcs.pixel_to_world(49.5, 49.5)
    corner = wcs.pixel_to_world(1, 1)
    assert moc.contains_lonlat(center.ra, center.dec)[0]
    assert moc.contains_lonlat(corner.ra, corner.dec)[0]
    outside = center.spherical_offsets_by(2 * u.deg, 0 * u.deg)
    assert not moc.contains_lonlat(outside.ra, outside.dec)[0]


def test_from_images_structure(image_catalog, images_df):
    assert isinstance(image_catalog, ImageCatalog)
    info = image_catalog.hc_structure.catalog_info
    assert info.catalog_type == CatalogType.IMAGE
    assert info.image_format == "fits"
    assert info.image_moc_order == MOC_ORDER
    computed = image_catalog.compute()
    assert set(computed["image_id"]) == set(images_df["image_id"])
    assert "band" in computed.columns
    # Footprints are not stored; WCS is a compact parameter struct
    assert "footprint_ranges" not in computed.columns
    assert "wcs" in computed.columns


def test_partition_contents_match_footprints(image_catalog, images_df, wcs_list):
    footprint_ranges = {
        image_id: np.asarray(image_footprint_moc(wcs, 100, 100, MOC_ORDER).to_depth29_ranges).astype(np.int64)
        for image_id, wcs in zip(images_df["image_id"], wcs_list)
    }
    for pixel in image_catalog.get_healpix_pixels():
        partition = image_catalog.get_partition(pixel.order, pixel.pixel).compute()
        shift = 2 * (SPATIAL_INDEX_ORDER - pixel.order)
        start, end = pixel.pixel << shift, (pixel.pixel + 1) << shift
        expected = {
            image_id
            for image_id, ranges in footprint_ranges.items()
            if np.any((ranges[:, 0] < end) & (ranges[:, 1] > start))
        }
        assert set(partition["image_id"]) == expected
        assert len(expected) > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_threshold_splits_partitions(images_df, wcs_list):
    coarse = from_images(images_df, wcs_list, threshold=100, lowest_order=0, highest_order=8)
    fine = from_images(images_df, wcs_list, threshold=1, lowest_order=0, highest_order=8)
    max_coarse = max(len(p) for p in coarse._operation.map_kwargs["df"])
    # With threshold=1, the overlapping pair v1/v2 forces splitting to deeper orders
    assert max(pixel.order for pixel in fine.get_healpix_pixels()) > max(
        pixel.order for pixel in coarse.get_healpix_pixels()
    )
    assert max_coarse <= 100


def test_threshold_warns_at_highest_order(images_df, wcs_list):
    # v1 and v2 overlap, so threshold=1 cannot be honored where they intersect
    with pytest.warns(UserWarning, match="more than the"):
        from_images(images_df, wcs_list, threshold=1, lowest_order=0, highest_order=2, moc_order=MOC_ORDER)


def test_wcs_params_column(image_catalog, wcs_list):
    computed = image_catalog.compute()
    params = computed["wcs"].iloc[0]
    assert set(params) == {
        "ctype1",
        "ctype2",
        "crval1",
        "crval2",
        "crpix1",
        "crpix2",
        "cd1_1",
        "cd1_2",
        "cd2_1",
        "cd2_2",
        "extra",
    }
    # Plain TAN WCS needs no header fallback
    assert params["extra"] is None
    # Parameters reconstruct the exact transform
    restored = wcs_from_params(params)
    row = computed.iloc[0]
    original = next(w for w in wcs_list if abs(w.wcs.crval[0] - row["wcs"]["crval1"]) < 1e-9)
    separation = original.pixel_to_world(10, 20).separation(restored.pixel_to_world(10, 20))
    assert separation.arcsec < 1e-9


def test_moc_is_union_of_footprints(image_catalog, wcs_list):
    moc = image_catalog.hc_structure.moc
    assert moc is not None
    for wcs in wcs_list:
        center = wcs.pixel_to_world(49.5, 49.5)
        assert moc.contains_lonlat(center.ra, center.dec)[0]


def test_wcs_for_row(image_catalog):
    computed = image_catalog.compute()
    row = computed[computed["image_id"] == "v1"].iloc[0]
    wcs = image_catalog.wcs_for(row)
    center = wcs.pixel_to_world(49.5, 49.5)
    assert abs(center.ra.deg - row["ra"]) < 1e-9
    assert abs(center.dec.deg - row["dec"]) < 1e-9


def test_from_images_validation(images_df, wcs_list):
    with pytest.raises(ValueError, match="wcs"):
        from_images(images_df)
    with pytest.raises(ValueError, match="3 images"):
        from_images(images_df, wcs_list[:2])
    with pytest.raises(ValueError, match="missing required columns"):
        from_images(images_df.drop(columns=["path"]), wcs_list)
    with pytest.raises(ValueError, match="must not exceed moc_order"):
        from_images(images_df, wcs_list, highest_order=12, moc_order=11)


def test_write_and_read_hats_round_trip(tmp_path, image_catalog):
    path = tmp_path / "demo_images"
    image_catalog.write_catalog(path)
    read_back = lsdb.read_hats(path)
    assert isinstance(read_back, ImageCatalog)
    assert read_back.hc_structure.catalog_info.catalog_type == CatalogType.IMAGE
    assert read_back.hc_structure.catalog_info.image_moc_order == MOC_ORDER
    assert read_back.get_healpix_pixels() == image_catalog.get_healpix_pixels()
    original = image_catalog.compute()
    restored = read_back.compute()
    assert len(restored) == len(original)
    assert set(restored["image_id"]) == set(original["image_id"])
    wcs = wcs_from_params(restored["wcs"].iloc[0])
    assert wcs.has_celestial
    # open_catalog works too
    opened = lsdb.open_catalog(path)
    assert isinstance(opened, ImageCatalog)


def test_wcs_params_distortion_fallback():
    from astropy.io import fits as pyfits

    header = pyfits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
    header["CRVAL1"], header["CRVAL2"] = 180.0, -30.0
    header["CRPIX1"], header["CRPIX2"] = 50.0, 50.0
    header["CD1_1"], header["CD1_2"] = -0.005, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, 0.005
    header["A_ORDER"], header["B_ORDER"] = 2, 2
    header["A_2_0"], header["B_0_2"] = 1e-5, 1e-5
    distorted = WCS(header)
    assert distorted.has_distortion

    params = wcs_to_params(distorted)
    assert params["extra"] is not None  # falls back to the full header string
    restored = wcs_from_params(params)
    assert restored.sip is not None
    separation = distorted.pixel_to_world(90, 90).separation(restored.pixel_to_world(90, 90))
    assert separation.arcsec < 1e-9
