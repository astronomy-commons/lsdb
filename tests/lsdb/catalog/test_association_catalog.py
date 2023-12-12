import lsdb
from lsdb.catalog.association_catalog import AssociationCatalog


def test_load_association(small_sky_to_xmatch_dir):
    small_sky_to_xmatch = lsdb.read_hipscat(small_sky_to_xmatch_dir)
    assert isinstance(small_sky_to_xmatch, AssociationCatalog)


def test_load_soft_association(small_sky_to_xmatch_soft_dir):
    small_sky_to_xmatch_soft = lsdb.read_hipscat(small_sky_to_xmatch_soft_dir)
    assert isinstance(small_sky_to_xmatch_soft, AssociationCatalog)
