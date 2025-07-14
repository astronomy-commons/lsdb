from importlib.metadata import version
from pathlib import Path

import hats as hc
import pandas as pd
from hats.io.paths import get_data_thumbnail_pointer

import lsdb
import lsdb.nested as nd
from lsdb.catalog.margin_catalog import MarginCatalog


def test_read_margin_catalog(small_sky_xmatch_margin_dir):
    margin = lsdb.open_catalog(small_sky_xmatch_margin_dir)
    assert isinstance(margin, MarginCatalog)
    assert isinstance(margin._ddf, nd.NestedFrame)
    hc_margin = hc.read_hats(small_sky_xmatch_margin_dir)
    assert margin.hc_structure.catalog_info == hc_margin.catalog_info
    assert margin.hc_structure.get_healpix_pixels() == hc_margin.get_healpix_pixels()
    assert margin.get_healpix_pixels() == margin.hc_structure.get_healpix_pixels()
    assert repr(margin) == repr(margin._ddf)
    pd.testing.assert_frame_equal(margin.compute(), margin._ddf.compute())


def test_margin_catalog_partitions_correct(small_sky_xmatch_margin_dir):
    margin = lsdb.open_catalog(small_sky_xmatch_margin_dir)
    assert isinstance(margin, MarginCatalog)
    for healpix_pixel in margin.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        path = hc.io.paths.pixel_catalog_file(
            catalog_base_dir=small_sky_xmatch_margin_dir,
            pixel=healpix_pixel,
        )
        partition = margin.get_partition(hp_order, hp_pixel)
        data = pd.read_parquet(path, dtype_backend="pyarrow").set_index("_healpix_29")
        pd.testing.assert_frame_equal(partition.compute(), data)


def test_save_margin_catalog(small_sky_xmatch_margin_catalog, tmp_path, helpers):
    new_catalog_name = "small_sky_xmatch_margin"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_xmatch_margin_catalog.to_hats(base_catalog_path, catalog_name=new_catalog_name)

    expected_catalog = lsdb.open_catalog(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == small_sky_xmatch_margin_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_xmatch_margin_catalog._ddf.compute())

    # When saving a catalog with to_hats, we update the hats_max_rows
    # to the maximum count of points per partition.
    partition_sizes = small_sky_xmatch_margin_catalog._ddf.map_partitions(len).compute()
    assert max(partition_sizes) == 10

    helpers.assert_catalog_info_is_correct(
        expected_catalog.hc_structure.catalog_info,
        small_sky_xmatch_margin_catalog.hc_structure.catalog_info,
        hats_max_rows="10",
        # Also check that the builder was properly set
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )

    # Sneak in test on data thumbnails: only main catalogs have them
    data_thumbnail_pointer = get_data_thumbnail_pointer(base_catalog_path)
    assert not data_thumbnail_pointer.exists()
