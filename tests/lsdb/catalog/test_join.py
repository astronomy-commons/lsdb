import nested_pandas as npd
import pandas as pd
import pyarrow as pa
import pytest
from hats.io import paths
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix

import lsdb.nested as nd
from lsdb import open_catalog
from lsdb.dask.merge_catalog_functions import align_catalogs


def test_small_sky_join_small_sky_order1(small_sky_catalog, small_sky_order1_catalog, helpers):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_order1_catalog, left_on="id", right_on="id", suffixes=suffixes
        )
        assert isinstance(joined._ddf, nd.NestedFrame)
    helpers.assert_columns_in_joined_catalog(joined, [small_sky_catalog, small_sky_order1_catalog], suffixes)
    assert joined._ddf.index.name == SPATIAL_INDEX_COLUMN
    assert joined._ddf.index.dtype == pd.ArrowDtype(pa.int64())
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    assert isinstance(joined_compute, npd.NestedFrame)
    small_sky_compute = small_sky_catalog.compute()
    small_sky_order1_compute = small_sky_order1_catalog.compute()
    assert len(joined_compute) == len(small_sky_compute)
    assert len(joined_compute) == len(small_sky_order1_compute)
    for index, row in small_sky_compute.iterrows():
        joined_row = joined_compute.query(f"id{suffixes[0]} == {row['id']}")
        assert joined_row.index.to_numpy()[0] == index
        assert joined_row[f"id{suffixes[1]}"].to_numpy()[0] == row["id"]
    helpers.assert_divisions_are_correct(joined)
    helpers.assert_schema_correct(joined)
    assert not joined.hc_structure.on_disk


def test_small_sky_join_small_sky_order1_source(
    small_sky_catalog, small_sky_order1_source_with_margin, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", suffixes=suffixes
    )
    helpers.assert_columns_in_joined_catalog(
        joined, [small_sky_catalog, small_sky_order1_source_with_margin], suffixes
    )

    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    small_sky_order1_compute = small_sky_order1_source_with_margin.compute()
    assert len(joined_compute) == len(small_sky_order1_compute)
    joined_test = small_sky_order1_compute.merge(joined_compute, left_on="object_id", right_on="object_id_b")
    assert (joined_test["id_a"].to_numpy() == joined_test["object_id"].to_numpy()).all()
    helpers.assert_divisions_are_correct(joined)
    helpers.assert_schema_correct(joined)


def test_small_sky_join_default_columns(
    small_sky_order1_default_cols_catalog, small_sky_order1_source_with_margin, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_order1_default_cols_catalog.join(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", suffixes=suffixes
    )
    helpers.assert_columns_in_joined_catalog(
        joined, [small_sky_order1_default_cols_catalog, small_sky_order1_source_with_margin], suffixes
    )

    alignment = align_catalogs(small_sky_order1_default_cols_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    small_sky_order1_compute = small_sky_order1_source_with_margin.compute()
    assert len(joined_compute) == len(small_sky_order1_compute)
    joined_test = small_sky_order1_compute.merge(joined_compute, left_on="object_id", right_on="object_id_b")
    assert (joined_test["id_a"].to_numpy() == joined_test["object_id"].to_numpy()).all()
    helpers.assert_divisions_are_correct(joined)
    helpers.assert_schema_correct(joined)
    helpers.assert_default_columns_in_columns(joined)


def test_join_wrong_columns(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="bad", right_on="id")
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="id", right_on="bad")


def test_join_wrong_suffixes(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="id", right_on="id", suffixes=("wrong",))


def test_join_association(
    small_sky_catalog, small_sky_order1_source_collection_catalog, small_sky_to_o1source_catalog, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog, suffixes=suffixes
    )
    assert isinstance(joined._ddf, nd.NestedFrame)
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_collection_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()
    helpers.assert_columns_in_joined_catalog(
        joined, [small_sky_catalog, small_sky_order1_source_collection_catalog], suffixes
    )
    assert joined._ddf.index.name == SPATIAL_INDEX_COLUMN
    assert joined._ddf.index.dtype == pd.ArrowDtype(pa.int64())

    joined_data = joined.compute()
    assert isinstance(joined_data, npd.NestedFrame)
    association_data = small_sky_to_o1source_catalog.compute()
    assert len(joined_data) == len(association_data)

    small_sky_compute = small_sky_catalog.compute()
    small_sky_xmatch_compute = small_sky_order1_source_collection_catalog.compute()

    for _, row in association_data.iterrows():
        left_col = small_sky_to_o1source_catalog.hc_structure.catalog_info.primary_column + suffixes[0]
        right_col = small_sky_to_o1source_catalog.hc_structure.catalog_info.join_column + suffixes[1]
        left_id = row[small_sky_to_o1source_catalog.hc_structure.catalog_info.primary_column_association]
        right_id = row[small_sky_to_o1source_catalog.hc_structure.catalog_info.join_column_association]
        joined_row = joined_data.query(f"{left_col} == {left_id} & {right_col} == {right_id}")
        assert len(joined_row) == 1

        small_sky_col = small_sky_to_o1source_catalog.hc_structure.catalog_info.primary_column
        left_row = small_sky_compute.query(f"{small_sky_col} == {left_id}")
        for col in left_row.columns:
            assert joined_row[col + suffixes[0]].to_numpy() == left_row[col].to_numpy()

        small_sky_xmatch_col = small_sky_to_o1source_catalog.hc_structure.catalog_info.join_column
        right_row = small_sky_xmatch_compute.query(f"{small_sky_xmatch_col} == {right_id}")
        for col in right_row.columns:
            assert joined_row[col + suffixes[1]].to_numpy() == right_row[col].to_numpy()

        assert joined_row.index == left_row.index


def test_join_association_suffix_edge_case(
    small_sky_catalog, small_sky_order1_source_collection_catalog, small_sky_to_o1source_catalog
):
    # Edge case: handle merge when right_column + suffix == join_column_association
    right_column = "source_id"
    suffix = small_sky_order1_source_collection_catalog.name
    join_column_assoc = small_sky_to_o1source_catalog.hc_structure.catalog_info.join_column_association
    assert f"{right_column}_{suffix}" == join_column_assoc

    xmatch_df = small_sky_catalog.crossmatch(
        small_sky_order1_source_collection_catalog, radius_arcsec=3600
    ).compute()

    join_df = small_sky_catalog.join(
        small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog
    ).compute()

    assert f"{right_column}_{suffix}" in join_df
    assert set(xmatch_df.columns) == set(join_df.columns)
    pd.testing.assert_frame_equal(xmatch_df, join_df, check_like=True)


def test_join_association_warnings(
    small_sky_catalog, small_sky_order1_source_collection_catalog, small_sky_to_o1source_catalog
):
    # Right catalog margin threshold < association max separation
    assert small_sky_to_o1source_catalog.max_separation > 436
    small_sky_order1_source_collection_catalog.margin.hc_structure.catalog_info.margin_threshold = 435
    with pytest.warns(RuntimeWarning, match="smaller than association maximum separation"):
        small_sky_catalog.join(
            small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog
        )
    # Association max separation is None
    small_sky_to_o1source_catalog.hc_structure.catalog_info.assn_max_separation = None
    with pytest.warns(RuntimeWarning, match="specify maximum separation"):
        small_sky_catalog.join(
            small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog
        )
    # Right catalog margin is None
    small_sky_order1_source_collection_catalog.margin = None
    with pytest.warns(RuntimeWarning, match="margin cache"):
        small_sky_catalog.join(
            small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog
        )


def test_join_nested(small_sky_catalog, small_sky_order1_source_with_margin, helpers):
    joined = small_sky_catalog.join_nested(
        small_sky_order1_source_with_margin,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
    )
    helpers.assert_columns_in_nested_joined_catalog(
        joined, small_sky_catalog, small_sky_order1_source_with_margin, ["object_id"], "sources"
    )
    helpers.assert_divisions_are_correct(joined)
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    source_compute = small_sky_order1_source_with_margin.compute()
    assert isinstance(joined_compute, npd.NestedFrame)
    for _, row in joined_compute.iterrows():
        row_id = row["id"]
        pd.testing.assert_frame_equal(
            row["sources"].sort_values("source_ra").reset_index(drop=True),
            pd.DataFrame(source_compute[source_compute["object_id"] == row_id].set_index("object_id"))
            .sort_values("source_ra")
            .reset_index(drop=True)
            .drop(columns=[c for c in paths.HIVE_COLUMNS if c in source_compute.columns]),
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
def test_merge_asof(small_sky_catalog, small_sky_xmatch_catalog, direction, helpers):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.merge_asof(small_sky_xmatch_catalog, direction=direction, suffixes=suffixes)
    assert isinstance(joined._ddf, nd.NestedFrame)
    helpers.assert_divisions_are_correct(joined)
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    assert isinstance(joined_compute, npd.NestedFrame)

    drop_cols = [c for c in paths.HIVE_COLUMNS if c in small_sky_catalog.columns]
    small_sky_compute = (
        small_sky_catalog.compute()
        .drop(columns=drop_cols)
        .rename(columns={c: c + suffixes[0] for c in small_sky_catalog.columns})
    )
    order_1_partition = spatial_index_to_healpix(small_sky_compute.index.to_numpy(), 1)
    left_partitions = [
        small_sky_compute[order_1_partition == p.pixel] for p in small_sky_xmatch_catalog.get_healpix_pixels()
    ]
    small_sky_order1_partitions = [
        p.compute()
        .drop(columns=drop_cols)
        .rename(columns={c: c + suffixes[1] for c in small_sky_xmatch_catalog.columns})
        for p in small_sky_xmatch_catalog.partitions
    ]
    correct_result = pd.concat(
        [
            pd.merge_asof(lp, rp, direction=direction, left_index=True, right_index=True)
            for lp, rp in zip(left_partitions, small_sky_order1_partitions)
        ]
    )
    pd.testing.assert_frame_equal(joined_compute.drop(columns=drop_cols), correct_result)


def merging_function(input_frame, map_input, *args, **kwargs):
    if len(input_frame) == 0:
        ## this is the empty call to infer meta
        return input_frame
    assert len(input_frame) == 131
    assert len(map_input) == 1
    assert args[0] == HealpixPixel(0, 11)
    assert args[1] == HealpixPixel(0, 11)
    assert kwargs == {"unused_kwarg": "ignored"}
    return input_frame


def test_merge_map(small_sky_catalog, test_data_dir):
    map_catalog = open_catalog(test_data_dir / "square_map")
    merge_lazy = small_sky_catalog.merge_map(map_catalog, merging_function, unused_kwarg="ignored")

    merge_result = merge_lazy.compute()
    assert len(merge_result) == small_sky_catalog.hc_structure.catalog_info.total_rows
