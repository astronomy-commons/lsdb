import hats as hc
import pandas as pd
import pytest

import lsdb
import lsdb.nested as nd
from lsdb.catalog.association_catalog import AssociationCatalog


@pytest.mark.skip("Re-implementing JOIN THROUGH")
def test_load_association(small_sky_to_xmatch_dir):
    small_sky_to_xmatch = lsdb.open_catalog(small_sky_to_xmatch_dir)
    assert isinstance(small_sky_to_xmatch, AssociationCatalog)
    assert isinstance(small_sky_to_xmatch._ddf, nd.NestedFrame)
    assert small_sky_to_xmatch.get_healpix_pixels() == small_sky_to_xmatch.hc_structure.get_healpix_pixels()
    assert repr(small_sky_to_xmatch) == repr(small_sky_to_xmatch._ddf)
    for healpix_pixel in small_sky_to_xmatch.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        path = hc.io.paths.pixel_catalog_file(
            catalog_base_dir=small_sky_to_xmatch_dir,
            pixel=healpix_pixel,
        )
        partition = small_sky_to_xmatch.get_partition(hp_order, hp_pixel)
        data = pd.read_parquet(path, dtype_backend="pyarrow")
        pd.testing.assert_frame_equal(partition.compute(), data)
