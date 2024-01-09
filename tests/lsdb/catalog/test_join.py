import pandas as pd
import pytest


def test_small_sky_join_small_sky_order1(
    small_sky_catalog, small_sky_order1_catalog, assert_divisions_are_correct
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(small_sky_order1_catalog, left_on="id", right_on="id", suffixes=suffixes)
    for col_name, dtype in small_sky_catalog.dtypes.items():
        assert (col_name + suffixes[0], dtype) in joined.dtypes.items()
    for col_name, dtype in small_sky_order1_catalog.dtypes.items():
        assert (col_name + suffixes[1], dtype) in joined.dtypes.items()
    joined_compute = joined.compute()
    small_sky_compute = small_sky_catalog.compute()
    small_sky_order1_compute = small_sky_order1_catalog.compute()
    assert len(joined_compute) == len(small_sky_compute)
    assert len(joined_compute) == len(small_sky_order1_compute)
    for index, row in small_sky_compute.iterrows():
        joined_row = joined_compute.query(f"id{suffixes[0]} == {row['id']}")
        assert joined_row.index.values[0] == index
        assert joined_row[f"id{suffixes[1]}"].values[0] == row["id"]
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
    joined = small_sky_catalog.join(
        small_sky_xmatch_catalog, through=small_sky_to_xmatch_catalog, suffixes=suffixes
    )
    assert joined._ddf.npartitions == len(small_sky_to_xmatch_catalog.hc_structure.join_info.data_frame)
    joined_data = joined.compute()
    association_data = small_sky_to_xmatch_catalog.compute()
    assert len(joined_data) == len(association_data)

    for col in small_sky_catalog._ddf.columns:
        assert col + suffixes[0] in joined._ddf.columns

    for col in small_sky_xmatch_catalog._ddf.columns:
        assert col + suffixes[1] in joined._ddf.columns

    for _, row in association_data.iterrows():
        left_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column + suffixes[0]
        right_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column + suffixes[1]
        left_id = row[small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column_association]
        right_id = row[small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column_association]
        joined_row = joined_data.query(f"{left_col} == {left_id} & {right_col} == {right_id}")
        assert len(joined_row) == 1
        small_sky_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.primary_column
        left_row = small_sky_catalog.compute().query(f"{small_sky_col}=={left_id}")
        for col in left_row.columns:
            assert joined_row[col + suffixes[0]].values == left_row[col].values

        small_sky_xmatch_col = small_sky_to_xmatch_catalog.hc_structure.catalog_info.join_column
        right_row = small_sky_xmatch_catalog.compute().query(f"{small_sky_xmatch_col}=={right_id}")
        for col in right_row.columns:
            assert joined_row[col + suffixes[1]].values == right_row[col].values

        left_index = left_row.index
        assert joined_row.index == left_index


def test_join_association_soft(small_sky_catalog, small_sky_xmatch_catalog, small_sky_to_xmatch_soft_catalog):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_xmatch_catalog, through=small_sky_to_xmatch_soft_catalog, suffixes=suffixes
    )
    assert joined._ddf.npartitions == len(small_sky_to_xmatch_soft_catalog.hc_structure.join_info.data_frame)

    joined_on = small_sky_catalog.join(
        small_sky_xmatch_catalog,
        left_on=small_sky_to_xmatch_soft_catalog.hc_structure.catalog_info.primary_column,
        right_on=small_sky_to_xmatch_soft_catalog.hc_structure.catalog_info.join_column,
        suffixes=suffixes,
    )
    pd.testing.assert_frame_equal(joined.compute(), joined_on.compute())
