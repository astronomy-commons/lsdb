import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix

from lsdb.dask.merge_catalog_functions import align_catalogs


def test_small_sky_join_small_sky_order1(
    small_sky_catalog, small_sky_order1_catalog, assert_divisions_are_correct
):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_order1_catalog, left_on="id", right_on="id", suffixes=suffixes
        )
        assert isinstance(joined._ddf, nd.NestedFrame)
    for col_name, dtype in small_sky_catalog.dtypes.items():
        assert (col_name + suffixes[0], dtype) in joined.dtypes.items()
    for col_name, dtype in small_sky_order1_catalog.dtypes.items():
        assert (col_name + suffixes[1], dtype) in joined.dtypes.items()
    assert joined._ddf.index.name == SPATIAL_INDEX_COLUMN
    assert joined._ddf.index.dtype == np.int64
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
    assert_divisions_are_correct(joined)


def test_small_sky_join_small_sky_order1_source(
    small_sky_catalog, small_sky_order1_source_with_margin, assert_divisions_are_correct
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", suffixes=suffixes
    )
    for col_name, dtype in small_sky_catalog.dtypes.items():
        assert (col_name + suffixes[0], dtype) in joined.dtypes.items()
    for col_name, dtype in small_sky_order1_source_with_margin.dtypes.items():
        assert (col_name + suffixes[1], dtype) in joined.dtypes.items()

    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    small_sky_order1_compute = small_sky_order1_source_with_margin.compute()
    assert len(joined_compute) == len(small_sky_order1_compute)
    joined_test = small_sky_order1_compute.merge(joined_compute, left_on="object_id", right_on="object_id_b")
    assert (joined_test["id_a"].to_numpy() == joined_test["object_id"].to_numpy()).all()
    assert_divisions_are_correct(joined)


def test_join_wrong_columns(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="bad", right_on="id")
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="id", right_on="bad")


def test_join_wrong_suffixes(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError):
        small_sky_catalog.join(small_sky_order1_catalog, left_on="id", right_on="id", suffixes=("wrong",))


def test_join_association(small_sky_catalog, small_sky_xmatch_catalog, small_sky_to_xmatch_catalog):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_xmatch_catalog, through=small_sky_to_xmatch_catalog, suffixes=suffixes
        )
        assert isinstance(joined._ddf, nd.NestedFrame)
    assert joined._ddf.npartitions == len(small_sky_to_xmatch_catalog.hc_structure.join_info.data_frame)
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_data = joined.compute()
    assert isinstance(joined_data, npd.NestedFrame)
    association_data = small_sky_to_xmatch_catalog.compute()
    assert len(joined_data) == len(association_data)

    for col in small_sky_catalog._ddf.columns:
        assert col + suffixes[0] in joined._ddf.columns
    for col in small_sky_xmatch_catalog._ddf.columns:
        assert col + suffixes[1] in joined._ddf.columns
    assert joined._ddf.index.name == SPATIAL_INDEX_COLUMN
    assert joined._ddf.index.dtype == np.int64

    small_sky_compute = small_sky_catalog.compute()
    small_sky_xmatch_compute = small_sky_xmatch_catalog.compute()

    for _, row in association_data.iterrows():
        left_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column + suffixes[0]
        right_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column + suffixes[1]
        left_id = row[small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column_association]
        right_id = row[small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column_association]
        joined_row = joined_data.query(f"{left_col} == {left_id} & {right_col} == {right_id}")
        assert len(joined_row) == 1
        small_sky_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column
        left_row = small_sky_compute.query(f"{small_sky_col}=={left_id}")
        for col in left_row.columns:
            assert joined_row[col + suffixes[0]].to_numpy() == left_row[col].to_numpy()

        small_sky_xmatch_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column
        right_row = small_sky_xmatch_compute.query(f"{small_sky_xmatch_col}=={right_id}")
        for col in right_row.columns:
            assert joined_row[col + suffixes[1]].to_numpy() == right_row[col].to_numpy()

        left_index = left_row.index
        assert joined_row.index == left_index


def test_join_association_source_margin(
    small_sky_catalog, small_sky_order1_source_with_margin, small_sky_to_o1source_catalog
):
    """Join the small sky object catalog to the order1 source catalog, including the margin
    of the order1 source catalog."""
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_with_margin, through=small_sky_to_o1source_catalog, suffixes=suffixes
    )
    assert joined._ddf.npartitions == len(small_sky_to_o1source_catalog.hc_structure.join_info.data_frame)
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_data = joined.compute()
    association_data = small_sky_to_o1source_catalog.compute()
    assert len(joined_data) == 17161
    assert len(association_data) == 17161

    assert (
        np.sort(joined_data["id_a"].to_numpy())
        == np.sort(
            association_data[
                small_sky_to_o1source_catalog.hc_structure.catalog_info.primary_column_association
            ].to_numpy()
        )
    ).all()
    assert (
        np.sort(joined_data["source_id_b"].to_numpy())
        == np.sort(
            association_data[
                small_sky_to_o1source_catalog.hc_structure.catalog_info.join_column_association
            ].to_numpy()
        )
    ).all()

    for col in small_sky_catalog._ddf.columns:
        assert col + suffixes[0] in joined._ddf.columns

    for col in small_sky_order1_source_with_margin._ddf.columns:
        assert col + suffixes[1] in joined._ddf.columns


def test_join_association_soft(small_sky_catalog, small_sky_xmatch_catalog, small_sky_to_xmatch_soft_catalog):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_xmatch_catalog, through=small_sky_to_xmatch_soft_catalog, suffixes=suffixes
        )
    assert joined._ddf.npartitions == len(small_sky_to_xmatch_soft_catalog.hc_structure.join_info.data_frame)
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    with pytest.warns(match="margin"):
        joined_on = small_sky_catalog.join(
            small_sky_xmatch_catalog,
            left_on=small_sky_to_xmatch_soft_catalog.hc_structure.catalog_info.primary_column,
            right_on=small_sky_to_xmatch_soft_catalog.hc_structure.catalog_info.join_column,
            suffixes=suffixes,
        )
    pd.testing.assert_frame_equal(joined.compute(), joined_on.compute())


def test_join_source_margin_soft(
    small_sky_catalog, small_sky_order1_source_with_margin, small_sky_to_o1source_soft_catalog
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_with_margin, through=small_sky_to_o1source_soft_catalog, suffixes=suffixes
    )
    assert joined._ddf.npartitions == len(
        small_sky_to_o1source_soft_catalog.hc_structure.join_info.data_frame
    )
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_on = small_sky_catalog.join(
        small_sky_order1_source_with_margin,
        left_on=small_sky_to_o1source_soft_catalog.hc_structure.catalog_info.primary_column,
        right_on=small_sky_to_o1source_soft_catalog.hc_structure.catalog_info.join_column,
        suffixes=suffixes,
    )
    pd.testing.assert_frame_equal(joined.compute(), joined_on.compute())


def test_join_nested(small_sky_catalog, small_sky_order1_source_with_margin, assert_divisions_are_correct):
    joined = small_sky_catalog.join_nested(
        small_sky_order1_source_with_margin,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
    )
    for col_name, dtype in small_sky_catalog.dtypes.items():
        assert (col_name, dtype) in joined.dtypes.items()
    for col_name, dtype in small_sky_order1_source_with_margin.dtypes.items():
        if col_name != "object_id":
            assert (col_name, dtype.pyarrow_dtype) in joined["sources"].dtypes.fields.items()
    assert_divisions_are_correct(joined)
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
            .reset_index(drop=True),
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )


def test_merge_asof(small_sky_catalog, small_sky_xmatch_catalog, assert_divisions_are_correct):
    suffixes = ("_a", "_b")
    for direction in ["backward", "forward", "nearest"]:
        joined = small_sky_catalog.merge_asof(
            small_sky_xmatch_catalog, direction=direction, suffixes=suffixes
        )
        assert isinstance(joined._ddf, nd.NestedFrame)
        assert_divisions_are_correct(joined)
        alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
        assert joined.hc_structure.moc == alignment.moc
        assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

        joined_compute = joined.compute()
        assert isinstance(joined_compute, npd.NestedFrame)
        small_sky_compute = small_sky_catalog.compute().rename(
            columns={c: c + suffixes[0] for c in small_sky_catalog.columns}
        )
        order_1_partition = spatial_index_to_healpix(small_sky_compute.index.to_numpy(), 1)
        left_partitions = [
            small_sky_compute[order_1_partition == p.pixel]
            for p in small_sky_xmatch_catalog.get_healpix_pixels()
        ]
        small_sky_order1_partitions = [
            p.compute().rename(columns={c: c + suffixes[1] for c in small_sky_xmatch_catalog.columns})
            for p in small_sky_xmatch_catalog.partitions
        ]
        correct_result = pd.concat(
            [
                pd.merge_asof(lp, rp, direction=direction, left_index=True, right_index=True)
                for lp, rp in zip(left_partitions, small_sky_order1_partitions)
            ]
        )
        pd.testing.assert_frame_equal(joined_compute, correct_result)
