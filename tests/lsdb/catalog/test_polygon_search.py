import pytest
from conftest import assert_divisions_are_correct
from spherical_geometry.polygon import SingleSphericalPolygon


def test_polygon_search_filters_correct_points(small_sky_order1_catalog):
    ra, dec = [300, 300, 272, 272], [-50, -55, -55, -50]
    polygon = SingleSphericalPolygon.from_lonlat(ra, dec)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(polygon)
    for _, row in polygon_search_catalog.compute().iterrows():
        ra = row[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
        dec = row[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
        assert polygon.contains_lonlat(ra, dec)
    assert_divisions_are_correct(polygon_search_catalog)


def test_polygon_search_filters_partitions(small_sky_order1_catalog):
    ra, dec = [300, 300, 272, 272], [-50, -55, -55, -50]
    polygon = SingleSphericalPolygon.from_lonlat(ra, dec)
    hc_polygon_search = small_sky_order1_catalog.hc_structure.filter_by_polygon(polygon)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(polygon)
    assert len(hc_polygon_search.get_healpix_pixels()) == len(polygon_search_catalog.get_healpix_pixels())
    assert len(hc_polygon_search.get_healpix_pixels()) == polygon_search_catalog._ddf.npartitions
    for pixel in hc_polygon_search.get_healpix_pixels():
        assert pixel in polygon_search_catalog._ddf_pixel_map


def test_polygon_search_empty(small_sky_order1_catalog):
    ra, dec = [0, 1, 0], [0, 1, 2]
    polygon = SingleSphericalPolygon.from_lonlat(ra, dec)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(polygon)
    assert len(polygon_search_catalog.get_healpix_pixels()) == 0
    assert len(polygon_search_catalog.hc_structure.pixel_tree) == 1


def test_polygon_search_invalid_shape(small_sky_order1_catalog):
    # Polygon is not convex, so the shape is invalid
    ra, dec = [0, 1, 1, 0], [1, 0, 1, 0]
    polygon = SingleSphericalPolygon.from_lonlat(ra, dec)
    with pytest.raises(RuntimeError):
        small_sky_order1_catalog.polygon_search(polygon)
