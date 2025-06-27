import pytest
from hats.pixel_math import HealpixPixel

import lsdb
import lsdb.nested as nd
from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.io.to_association import _check_catalogs_and_columns, to_association


@pytest.fixture
def xmatch_result(small_sky_catalog, small_sky_xmatch_catalog):
    xmatch_result = lsdb.crossmatch(
        small_sky_catalog,
        small_sky_xmatch_catalog,
        suffixes=["_left", "_right"],
        radius_arcsec=0.01 * 3600,
        left_args={"margin_threshold": 100},
        right_args={"margin_threshold": 100},
    )[["id_left", "id_right"]]
    assert list(xmatch_result.columns) == ["id_left", "id_right"]
    return xmatch_result


@pytest.fixture
def association_kwargs(small_sky_dir, small_sky_xmatch_dir):
    return {
        "primary_catalog_dir": small_sky_dir,
        "primary_column_association": "id_left",
        "primary_id_column": "id",
        "join_catalog_dir": small_sky_xmatch_dir,
        "join_column_association": "id_right",
        "join_id_column": "id",
    }


def test_crossmatch_to_association(xmatch_result, association_kwargs, tmp_path):
    to_association(
        xmatch_result,
        catalog_name="test_association",
        base_catalog_path=tmp_path,
        overwrite=False,
        **association_kwargs,
    )
    association_table = lsdb.read_hats(tmp_path)
    assert isinstance(association_table, AssociationCatalog)
    assert isinstance(association_table._ddf, nd.NestedFrame)
    assert association_table.get_healpix_pixels() == [
        HealpixPixel(1, 44),
        HealpixPixel(1, 45),
        HealpixPixel(1, 46),
    ]


def test_to_association_overwrite(xmatch_result, association_kwargs, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    to_association(
        xmatch_result,
        catalog_name="test_association",
        base_catalog_path=base_catalog_path,
        overwrite=False,
        **association_kwargs,
    )
    # The output directory exists and it has content. Overwrite is
    # set to False and, as such, the operation fails.
    with pytest.raises(ValueError, match="set overwrite to True"):
        to_association(
            xmatch_result,
            catalog_name="test_association",
            base_catalog_path=base_catalog_path,
            overwrite=False,
            **association_kwargs,
        )
    # With overwrite it succeeds because the directory is recreated
    to_association(
        xmatch_result,
        catalog_name="test_association",
        base_catalog_path=base_catalog_path,
        overwrite=True,
        **association_kwargs,
    )


def test_association_required_column_names(association_kwargs, tmp_path):
    column_names = ["id_left", "id_right"]
    _check_catalogs_and_columns(column_names, **association_kwargs)

    ## Try setting each field to None, and be sad about it.
    for field in association_kwargs.keys():
        missing_field = association_kwargs | {field: None}
        with pytest.raises(ValueError, match=field):
            _check_catalogs_and_columns(column_names, **missing_field)

    ## Try setting association columns to ones that aren't in the catalog.
    for field in ["primary_column_association", "join_column_association"]:
        missing_field = association_kwargs | {field: "garbage"}
        with pytest.raises(ValueError, match=field):
            _check_catalogs_and_columns(column_names, **missing_field)

    ## Try setting catalog columns to ones that aren't in the original catalogs.
    for field in ["primary_id_column", "join_id_column", "join_to_primary_id_column"]:
        missing_field = association_kwargs | {field: "garbage"}
        with pytest.raises(ValueError, match=field):
            _check_catalogs_and_columns(column_names, **missing_field)

    ## Try passing in catalog directory that's not a valid catalog directory.
    for field in ["primary_catalog_dir", "join_catalog_dir"]:
        missing_field = association_kwargs | {field: tmp_path}
        with pytest.raises(FileNotFoundError):
            _check_catalogs_and_columns(column_names, **missing_field)


def test_association_with_collections(
    small_sky_order1_collection_dir, small_sky_order1_source_collection_dir
):
    """Confirm that catalog directories on either side can be collections.
    Confirm that with a join-to-primary column, everything is populated correctly."""
    column_names = ["OBJECT_ID", "SOURCE_ID"]

    returned_values = _check_catalogs_and_columns(
        column_names,
        **{
            "primary_catalog_dir": small_sky_order1_collection_dir,
            "primary_column_association": "OBJECT_ID",
            "primary_id_column": "id",
            "join_catalog_dir": small_sky_order1_source_collection_dir,
            "join_column_association": "SOURCE_ID",
            "join_id_column": "object_id",
            "join_to_primary_id_column": "source_id",
        },
    )

    assert returned_values == {
        "primary_column": "id",
        "primary_column_association": "OBJECT_ID",
        "primary_catalog": str(small_sky_order1_collection_dir),
        "join_column": "object_id",
        "join_column_association": "SOURCE_ID",
        "join_catalog": str(small_sky_order1_source_collection_dir),
        "join_to_primary_id_column": "source_id",
    }
