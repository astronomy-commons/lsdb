import lsdb
import lsdb.nested as nd
from lsdb import io
from lsdb.catalog.association_catalog import AssociationCatalog


def test_crossmatch_to_association(small_sky_catalog, small_sky_xmatch_catalog, tmp_path):
    # Perform the crossmatch
    xmatch_result = lsdb.crossmatch(
        small_sky_catalog,
        small_sky_xmatch_catalog,
        suffixes=["_left", "_right"],
        radius_arcsec=0.01 * 3600,
        left_args={"margin_threshold": 100},
        right_args={"margin_threshold": 100},
    )[["id_left", "id_right"]]
    assert list(xmatch_result.columns) == ["id_left", "id_right"]

    io.to_association(
        xmatch_result,
        base_catalog_path=tmp_path,
        catalog_name="test_association",
        primary_catalog_dir=small_sky_catalog.hc_structure.catalog_path,
        primary_column_association="id_left",
        primary_id_column="id",
        join_catalog_dir=small_sky_xmatch_catalog.hc_structure.catalog_path,
        join_column_association="id_right",
        join_id_column="id",
        overwrite=True,
    )
    association_table = lsdb.read_hats(tmp_path)
    assert isinstance(association_table, AssociationCatalog)
    assert isinstance(association_table._ddf, nd.NestedFrame)
    assert (
        association_table.get_healpix_pixels() == small_sky_xmatch_catalog.hc_structure.get_healpix_pixels()
    )
