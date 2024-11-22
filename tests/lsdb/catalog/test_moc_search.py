import astropy.units as u
import numpy as np
import pandas as pd
from hats.pixel_math import HealpixPixel
from mocpy import MOC


def test_moc_search_filters_correct_points(small_sky_order1_catalog):
    search_moc = MOC.from_healpix_cells(ipix=np.array([176, 177]), depth=np.array([2, 2]), max_depth=2)
    filtered_cat = small_sky_order1_catalog.moc_search(search_moc)
    assert filtered_cat.get_healpix_pixels() == [HealpixPixel(1, 44)]
    filtered_cat_comp = filtered_cat.compute()
    cat_comp = small_sky_order1_catalog.compute()
    assert np.all(
        search_moc.contains_lonlat(
            filtered_cat_comp["ra"].to_numpy() * u.deg, filtered_cat_comp["dec"].to_numpy() * u.deg
        )
    )
    assert np.sum(
        search_moc.contains_lonlat(cat_comp["ra"].to_numpy() * u.deg, cat_comp["dec"].to_numpy() * u.deg)
    ) == len(filtered_cat_comp)


def test_moc_search_non_fine(small_sky_order1_catalog):
    search_moc = MOC.from_healpix_cells(ipix=np.array([176, 180]), depth=np.array([2, 2]), max_depth=2)
    filtered_cat = small_sky_order1_catalog.moc_search(search_moc, fine=False)
    assert filtered_cat.get_healpix_pixels() == [HealpixPixel(1, 44), HealpixPixel(1, 45)]
    pd.testing.assert_frame_equal(
        filtered_cat.compute(),
        small_sky_order1_catalog.pixel_search([HealpixPixel(1, 44), HealpixPixel(1, 45)]).compute(),
    )
