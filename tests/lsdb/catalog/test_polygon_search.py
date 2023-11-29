import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion


def test_polygon_search_filters_correct_points(small_sky_order1_catalog):
    max_order = 1
    vertices = SkyCoord(ra=[252, 252, 272, 272], dec=[-58, -55, -55, -58], unit="deg")
    polygon = PolygonSkyRegion(vertices=vertices)

    polygon_search_df = small_sky_order1_catalog.polygon_search(polygon).compute()

    filtered_pixels = hp.ang2pix(
        2**max_order,
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column],
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column],
        lonlat=True,
        nest=True,
    )
    polygon_pixels = hp.query_polygon(
        hp.order2nside(max_order),
        np.array(polygon.vertices.cartesian.xyz).T,
        inclusive=True,
        nest=True
    )
    assert np.all(np.isin(filtered_pixels, polygon_pixels))


def test_polygon_search_filters_partitions(small_sky_order1_catalog):
    vertices = SkyCoord(ra=[252, 252, 272, 272], dec=[-58, -55, -55, -58], unit="deg")
    polygon = PolygonSkyRegion(vertices=vertices)

    hc_polygon_search, _ = small_sky_order1_catalog.hc_structure.filter_by_polygon(polygon)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(polygon)

    assert len(hc_polygon_search.get_healpix_pixels()) == len(polygon_search_catalog.get_healpix_pixels())
    assert len(hc_polygon_search.get_healpix_pixels()) == polygon_search_catalog._ddf.npartitions
    for pixel in hc_polygon_search.get_healpix_pixels():
        assert pixel in polygon_search_catalog._ddf_pixel_map


def test_polygon_search_empty(small_sky_order1_catalog):
    vertices = SkyCoord([0, 1, 0], [0, 1, 2], unit="deg")
    polygon = PolygonSkyRegion(vertices=vertices)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(polygon)
    assert len(polygon_search_catalog.get_healpix_pixels()) == 0
    assert len(polygon_search_catalog.hc_structure.pixel_tree) == 1


def test_polygon_search_invalid_shape(small_sky_order1_catalog):
    # Polygon is not convex, so the shape is invalid
    vertices = SkyCoord([0, 1, 1, 0], [1, 0, 1, 0], unit="deg")
    polygon = PolygonSkyRegion(vertices=vertices)
    with pytest.raises(RuntimeError):
        small_sky_order1_catalog.polygon_search(polygon)
