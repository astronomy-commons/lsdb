import pytest
from astropy.coordinates import SkyCoord


def test_cone_search_filters_correct_points(small_sky_order1_catalog):
    ra = 0
    dec = -80
    radius = 20
    center_coord = SkyCoord(ra, dec, unit='deg')
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius).compute()
    print(len(cone_search_catalog))
    for _, row in small_sky_order1_catalog.compute().iterrows():
        row_ra = row[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
        row_dec = row[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
        sep = SkyCoord(row_ra, row_dec, unit='deg').separation(center_coord)
        if sep.degree <= radius:
            assert len(cone_search_catalog.loc[cone_search_catalog["id"] == row["id"]]) == 1
        else:
            assert len(cone_search_catalog.loc[cone_search_catalog["id"] == row["id"]]) == 0


def test_cone_search_filters_partitions(small_sky_order1_catalog):
    ra = 0
    dec = -80
    radius = 20
    hc_conesearch = small_sky_order1_catalog.hc_structure.filter_by_cone(ra, dec, radius)
    consearch_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius)
    assert len(hc_conesearch.get_healpix_pixels()) == len(consearch_catalog.hc_structure.get_pixels())
    assert len(hc_conesearch.get_healpix_pixels()) == consearch_catalog._ddf.npartitions
    print(hc_conesearch.get_healpix_pixels())
    for pixel in hc_conesearch.get_healpix_pixels():
        assert pixel in consearch_catalog._ddf_pixel_map


def test_negative_radius_errors(small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_order1_catalog.cone_search(0, 0, -1)


def test_invalid_ra_dec(small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_order1_catalog.cone_search(-200, 0, 1)
    with pytest.raises(ValueError):
        small_sky_order1_catalog.cone_search(200, 0, 1)
    with pytest.raises(ValueError):
        small_sky_order1_catalog.cone_search(0, -100, 1)
    with pytest.raises(ValueError):
        small_sky_order1_catalog.cone_search(0, 100, 1)
