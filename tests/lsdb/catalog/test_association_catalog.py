import hipscat as hc
import pandas as pd

import lsdb
from lsdb.catalog.association_catalog import AssociationCatalog


def test_load_association(small_sky_to_xmatch_dir):
    small_sky_to_xmatch = lsdb.read_hipscat(small_sky_to_xmatch_dir)
    assert isinstance(small_sky_to_xmatch, AssociationCatalog)
    assert small_sky_to_xmatch.get_healpix_pixels() == small_sky_to_xmatch.hc_structure.get_healpix_pixels()
    assert repr(small_sky_to_xmatch) == repr(small_sky_to_xmatch._ddf)
    for healpix_pixel in small_sky_to_xmatch.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        path = hc.io.paths.pixel_catalog_file(
            catalog_base_dir=small_sky_to_xmatch_dir,
            pixel_order=hp_order,
            pixel_number=hp_pixel,
        )
        partition = small_sky_to_xmatch.get_partition(hp_order, hp_pixel)
        data = pd.read_parquet(path)
        pd.testing.assert_frame_equal(partition.compute(), data)


def test_load_soft_association(small_sky_to_xmatch_soft_dir):
    small_sky_to_xmatch_soft = lsdb.read_hipscat(small_sky_to_xmatch_soft_dir)
    assert isinstance(small_sky_to_xmatch_soft, AssociationCatalog)
    assert len(small_sky_to_xmatch_soft.compute()) == 0
