from pathlib import Path

import hipscat as hc
import pandas as pd

import lsdb
from lsdb.catalog.margin_catalog import MarginCatalog


def test_read_margin_catalog(small_sky_xmatch_margin_dir):
    margin = lsdb.read_hipscat(small_sky_xmatch_margin_dir)
    assert isinstance(margin, MarginCatalog)
    hc_margin = hc.catalog.MarginCatalog.read_from_hipscat(small_sky_xmatch_margin_dir)
    assert margin.hc_structure.catalog_info == hc_margin.catalog_info
    assert margin.hc_structure.get_healpix_pixels() == hc_margin.get_healpix_pixels()
    assert margin.get_healpix_pixels() == margin.hc_structure.get_healpix_pixels()
    assert repr(margin) == repr(margin._ddf)
    pd.testing.assert_frame_equal(margin.compute(), margin._ddf.compute())


def test_margin_catalog_partitions_correct(small_sky_xmatch_margin_dir):
    margin = lsdb.read_hipscat(small_sky_xmatch_margin_dir)
    assert isinstance(margin, MarginCatalog)
    for healpix_pixel in margin.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        path = hc.io.paths.pixel_catalog_file(
            catalog_base_dir=small_sky_xmatch_margin_dir,
            pixel_order=hp_order,
            pixel_number=hp_pixel,
        )
        partition = margin.get_partition(hp_order, hp_pixel)
        data = pd.read_parquet(path)
        pd.testing.assert_frame_equal(partition.compute(), data)


def test_save_margin_catalog(small_sky_xmatch_margin_catalog, tmp_path):
    new_catalog_name = "small_sky_xmatch_margin"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_xmatch_margin_catalog.to_hipscat(base_catalog_path, catalog_name=new_catalog_name)
    expected_catalog = lsdb.read_hipscat(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert (
        expected_catalog.hc_structure.catalog_info
        == small_sky_xmatch_margin_catalog.hc_structure.catalog_info
    )
    assert expected_catalog.get_healpix_pixels() == small_sky_xmatch_margin_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_xmatch_margin_catalog._ddf.compute())
