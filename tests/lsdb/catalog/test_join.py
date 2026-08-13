import re

import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hats.io import paths
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix

import lsdb
from lsdb.operations.functions.merge_catalog_functions import align_catalogs

SPARSE_RIGHT_PIXELS = [HealpixPixel(1, 46), HealpixPixel(1, 47)]
CORE_ONLY_PIXELS = [HealpixPixel(1, 44), HealpixPixel(1, 45)]
MATCHED_OBJECT_IDS = {700, 756}


def test_small_sky_join_small_sky_order1(small_sky_catalog, small_sky_order1_catalog, helpers):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_order1_catalog, left_on="id", right_on="id", suffixes=suffixes
        )
        assert isinstance(joined.meta, npd.NestedFrame)

    expected_columns = [
        "id_a",
        "ra_a",
        "dec_a",
        "ra_error_a",
        "dec_error_a",
        "id_b",
        "ra_b",
        "dec_b",
        "ra_error_b",
        "dec_error_b",
    ]

    assert np.all(joined.columns == expected_columns)
    assert joined.meta.index.name == SPATIAL_INDEX_COLUMN
    assert joined.meta.index.dtype == pd.ArrowDtype(pa.int64())
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    assert np.all(joined_compute.columns == expected_columns)
    assert isinstance(joined_compute, npd.NestedFrame)
    small_sky_compute = small_sky_catalog.compute()
    small_sky_order1_compute = small_sky_order1_catalog.compute()
    assert len(joined_compute) == len(small_sky_compute)
    assert len(joined_compute) == len(small_sky_order1_compute)
    for index, row in small_sky_compute.iterrows():
        joined_row = joined_compute.query(f"id{suffixes[0]} == {row['id']}")
        assert joined_row.index.to_numpy()[0] == index
        assert joined_row[f"id{suffixes[1]}"].to_numpy()[0] == row["id"]
    helpers.assert_schema_correct(joined)
    assert not joined.hc_structure.on_disk
    assert joined.est_size() is None


def test_small_sky_join_overlapping_suffix(small_sky_catalog, small_sky_order1_catalog, helpers):
    suffixes = ("_a", "_b")
    with pytest.warns(match="margin"):
        joined = small_sky_catalog.join(
            small_sky_order1_catalog,
            left_on="id",
            right_on="id",
            suffixes=suffixes,
            suffix_method="overlapping_columns",
        )
        assert isinstance(joined.meta, npd.NestedFrame)

    expected_columns = [
        "id_a",
        "ra_a",
        "dec_a",
        "ra_error_a",
        "dec_error_a",
        "id_b",
        "ra_b",
        "dec_b",
        "ra_error_b",
        "dec_error_b",
    ]

    assert np.all(joined.columns == expected_columns)

    joined_compute = joined.compute()

    assert np.all(joined_compute.columns == expected_columns)

    helpers.assert_schema_correct(joined)


def test_small_sky_join_small_sky_order1_source(
    small_sky_catalog, small_sky_order1_source_with_margin, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", suffixes=suffixes
    )

    expected_columns = [
        "id_a",
        "ra_a",
        "dec_a",
        "ra_error_a",
        "dec_error_a",
        "source_id_b",
        "source_ra_b",
        "source_dec_b",
        "mjd_b",
        "mag_b",
        "band_b",
        "object_id_b",
        "object_ra_b",
        "object_dec_b",
    ]

    assert np.all(joined.columns == expected_columns)

    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()

    assert np.all(joined_compute.columns == expected_columns)
    small_sky_order1_compute = small_sky_order1_source_with_margin.compute()
    assert len(joined_compute) == len(small_sky_order1_compute)
    joined_test = small_sky_order1_compute.merge(joined_compute, left_on="object_id", right_on="object_id_b")
    assert (joined_test["id_a"].to_numpy() == joined_test["object_id"].to_numpy()).all()
    helpers.assert_schema_correct(joined)


def test_small_sky_join_default_columns(
    small_sky_order1_default_cols_catalog, small_sky_order1_source_with_margin, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_order1_default_cols_catalog.join(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", suffixes=suffixes
    )

    expected_columns = [
        "ra_a",
        "dec_a",
        "id_a",
        "source_id_b",
        "source_ra_b",
        "source_dec_b",
        "mjd_b",
        "mag_b",
        "band_b",
        "object_id_b",
        "object_ra_b",
        "object_dec_b",
    ]

    assert np.all(joined.columns == expected_columns)

    alignment = align_catalogs(small_sky_order1_default_cols_catalog, small_sky_order1_source_with_margin)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    joined_compute = joined.compute()
    assert np.all(joined_compute.columns == expected_columns)
    small_sky_order1_compute = small_sky_order1_source_with_margin.compute()
    assert len(joined_compute) == len(small_sky_order1_compute)
    joined_test = small_sky_order1_compute.merge(joined_compute, left_on="object_id", right_on="object_id_b")
    assert (joined_test["id_a"].to_numpy() == joined_test["object_id"].to_numpy()).all()
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
    small_sky_catalog, small_sky_order1_source_collection_catalog, small_sky_to_o1source_catalog
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_collection_catalog, through=small_sky_to_o1source_catalog, suffixes=suffixes
    )
    assert isinstance(joined.meta, npd.NestedFrame)
    alignment = align_catalogs(small_sky_catalog, small_sky_order1_source_collection_catalog)
    assert joined.hc_structure.moc == alignment.moc
    assert joined.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    expected_columns = [
        "id_a",
        "ra_a",
        "dec_a",
        "ra_error_a",
        "dec_error_a",
        "source_id_b",
        "source_ra_b",
        "source_dec_b",
        "mjd_b",
        "mag_b",
        "band_b",
        "object_id_b",
        "object_ra_b",
        "object_dec_b",
        "_dist_arcsec",
    ]

    assert np.all(joined.columns == expected_columns)

    assert joined.meta.index.name == SPATIAL_INDEX_COLUMN
    assert joined.meta.index.dtype == pd.ArrowDtype(pa.int64())

    joined_data = joined.compute()

    assert np.all(joined_data.columns == expected_columns)
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


def test_join_association_overlapping_suffix(
    small_sky_catalog, small_sky_order1_source_collection_catalog, small_sky_to_o1source_catalog, helpers
):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.join(
        small_sky_order1_source_collection_catalog,
        through=small_sky_to_o1source_catalog,
        suffixes=suffixes,
        suffix_method="overlapping_columns",
    )
    expected_columns = [
        "id",
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "source_id",
        "source_ra",
        "source_dec",
        "mjd",
        "mag",
        "band",
        "object_id",
        "object_ra",
        "object_dec",
        "_dist_arcsec",
    ]

    assert np.all(joined.columns == expected_columns)

    joined_compute = joined.compute()

    assert np.all(joined_compute.columns == expected_columns)

    helpers.assert_schema_correct(joined)


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


def test_join_nested(small_sky_catalog, small_sky_order1_source_with_margin):
    joined = small_sky_catalog.join_nested(
        small_sky_order1_source_with_margin,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
    )
    expected_columns = [
        "id",
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "sources",
    ]
    expected_nested_columns = [
        "source_id",
        "source_ra",
        "source_dec",
        "mjd",
        "mag",
        "band",
        "object_ra",
        "object_dec",
    ]
    assert np.all(joined.columns == expected_columns)
    assert np.all(joined.meta["sources"].nest.columns == expected_nested_columns)
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


def test_join_nested_how_left(small_sky_order1_catalog, small_sky_order1_source_with_margin, helpers):
    # All pixels in the object catalog have a corresponding source pixel
    object_pixels = small_sky_order1_catalog.get_healpix_pixels()
    source_pixels = small_sky_order1_source_with_margin.get_healpix_pixels()
    assert all(p in source_pixels for p in object_pixels)

    # Now we will select only two pixels from the source catalog
    selected_pixels = [HealpixPixel(1, 46), HealpixPixel(1, 47)]
    smaller_sky_sources = small_sky_order1_source_with_margin.pixel_search(selected_pixels, fine=True)

    # If we `join_nested` with `how="left"`, we keep all objects on the left
    nested_left = small_sky_order1_catalog.join_nested(
        smaller_sky_sources,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
        how="left",
    )
    helpers.assert_columns_in_nested_joined_catalog(
        nested_left, small_sky_order1_catalog, smaller_sky_sources, ["object_id"], "sources"
    )

    # All object pixels will show up in the final result
    assert object_pixels == nested_left.get_healpix_pixels()
    nested_left_compute = nested_left.compute()
    assert len(small_sky_order1_catalog) == len(nested_left_compute)

    source_compute = smaller_sky_sources.compute()
    source_margin = smaller_sky_sources.margin.compute()
    total_sources = pd.concat([source_compute, source_margin]).drop_duplicates(subset="source_id")
    for _, row in nested_left_compute.iterrows():
        row_id = row["id"]
        if row["sources"] is not None:
            pd.testing.assert_frame_equal(
                row["sources"].sort_values("source_ra").reset_index(drop=True),
                pd.DataFrame(total_sources[total_sources["object_id"] == row_id].set_index("object_id"))
                .sort_values("source_ra")
                .reset_index(drop=True)
                .drop(columns=[c for c in paths.HIVE_COLUMNS if c in source_compute.columns]),
                check_dtype=False,
                check_column_type=False,
                check_index_type=False,
            )


def test_join_nested_how_left_with_sparse_independent_storage(
    small_sky_order1_catalog, materialized_sparse_right_catalog, helpers
):
    right_store, sparse_right_catalog = materialized_sparse_right_catalog
    core_catalog_path = small_sky_order1_catalog.hc_structure.catalog_base_dir
    right_catalog_path = sparse_right_catalog.hc_structure.catalog_base_dir

    # The Extension stand-in is an ordinary, physically persisted HATS collection,
    # reopened from a different fsspec backend than the local Core fixture.
    assert (right_store / "collection.properties").is_file()
    assert sparse_right_catalog.hc_structure.on_disk
    assert sparse_right_catalog.margin is not None
    assert sparse_right_catalog.margin.hc_structure.on_disk
    assert core_catalog_path.protocol != right_catalog_path.protocol
    assert sparse_right_catalog.margin.hc_structure.catalog_base_dir.protocol == right_catalog_path.protocol
    assert (right_catalog_path / "hats.properties").is_file()
    assert (right_catalog_path / "partition_info.csv").is_file()

    # Only two right-catalog pixels exist physically; the other Core pixels are
    # scientific non-coverage rather than declared-but-missing storage objects.
    object_pixels = small_sky_order1_catalog.get_healpix_pixels()
    assert sparse_right_catalog.get_healpix_pixels() == SPARSE_RIGHT_PIXELS
    assert set(object_pixels) - set(SPARSE_RIGHT_PIXELS) == set(CORE_ONLY_PIXELS)
    assert all(paths.pixel_catalog_file(right_catalog_path, pixel).is_file() for pixel in SPARSE_RIGHT_PIXELS)
    assert all(not paths.pixel_catalog_file(right_catalog_path, pixel).exists() for pixel in CORE_ONLY_PIXELS)

    # If we `join_nested` with `how="left"`, we keep all objects on the left
    nested_left = small_sky_order1_catalog.join_nested(
        sparse_right_catalog,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
        how="left",
    )
    helpers.assert_columns_in_nested_joined_catalog(
        nested_left, small_sky_order1_catalog, sparse_right_catalog, ["object_id"], "sources"
    )

    # Sparse coverage must preserve every core pixel, row, index, and column.
    assert object_pixels == nested_left.get_healpix_pixels()
    nested_left_compute = nested_left.compute(progress_bar=False)
    core_compute = small_sky_order1_catalog.compute(progress_bar=False)
    pd.testing.assert_frame_equal(nested_left_compute.drop(columns="sources"), core_compute)

    source_compute = sparse_right_catalog.compute(progress_bar=False)
    source_margin = sparse_right_catalog.margin.compute(progress_bar=False)
    assert set(source_compute["object_id"].astype(int)) == MATCHED_OBJECT_IDS
    total_sources = pd.concat([source_compute, source_margin])
    scientific_columns = [column for column in total_sources.columns if column not in paths.HIVE_COLUMNS]
    total_sources = total_sources.loc[:, scientific_columns].drop_duplicates()
    assert total_sources["source_id"].is_unique
    observed_matches = set()
    observed_nulls = set()
    for _, row in nested_left_compute.iterrows():
        row_id = int(row["id"])
        sources = row["sources"]
        if sources is None:
            observed_nulls.add(row_id)
            continue

        observed_matches.add(row_id)
        assert sources["extension_object_id"].eq(row_id).all()
        assert sources["source_id"].is_unique
        pd.testing.assert_frame_equal(
            sources.sort_values("source_ra").reset_index(drop=True),
            pd.DataFrame(total_sources[total_sources["object_id"] == row_id].set_index("object_id"))
            .sort_values("source_ra")
            .reset_index(drop=True),
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )

    core_id_series = core_compute["id"].astype(int)
    assert core_id_series.is_unique
    core_ids = set(core_id_series)
    expected_nulls = core_ids - MATCHED_OBJECT_IDS
    assert observed_matches == MATCHED_OBJECT_IDS
    assert observed_nulls == expected_nulls
    assert len(observed_matches) + len(observed_nulls) == len(core_compute)


def test_join_nested_how_left_is_partition_local_for_matching_ids(small_sky_order1_catalog):
    # This characterizes the documented join_nested behavior only. It does not define
    # the future identity or partitioning contract for Core + Extension catalogs.
    core_compute = small_sky_order1_catalog.compute(progress_bar=False)
    target_row = core_compute.loc[core_compute["id"] == 700].iloc[0]
    wrong_pixel_row = core_compute.loc[core_compute["id"] == 756].iloc[0]
    target_pixel = HealpixPixel(
        1, int(spatial_index_to_healpix(np.array([target_row.name]), target_order=1)[0])
    )
    wrong_pixel = HealpixPixel(
        1, int(spatial_index_to_healpix(np.array([wrong_pixel_row.name]), target_order=1)[0])
    )
    assert target_pixel == HealpixPixel(1, 46)
    assert wrong_pixel == HealpixPixel(1, 47)

    displaced_extension = lsdb.from_dataframe(
        pd.DataFrame(
            {
                "extension_id": [7000],
                "object_id": [700],
                "ra": [wrong_pixel_row["ra"]],
                "dec": [wrong_pixel_row["dec"]],
            }
        ),
        lowest_order=1,
        highest_order=1,
        margin_threshold=None,
        should_generate_moc=False,
        catalog_name="displaced_extension",
    )
    assert displaced_extension.get_healpix_pixels() == [wrong_pixel]
    assert wrong_pixel in small_sky_order1_catalog.get_healpix_pixels()
    assert wrong_pixel != target_pixel

    displaced_compute = displaced_extension.compute(progress_bar=False)
    assert 700 in set(core_compute["id"].astype(int))
    assert 700 in set(displaced_compute["object_id"].astype(int))

    with pytest.warns(RuntimeWarning, match="margin cache"):
        nested_left = small_sky_order1_catalog.join_nested(
            displaced_extension,
            left_on="id",
            right_on="object_id",
            nested_column_name="extensions",
            how="left",
        )

    nested_left_compute = nested_left.compute(progress_bar=False)
    pd.testing.assert_frame_equal(nested_left_compute.drop(columns="extensions"), core_compute)
    assert nested_left_compute["extensions"].isna().all()


def test_join_nested_how_left_raises_when_declared_right_partition_is_missing(
    small_sky_order1_catalog,
    materialized_sparse_right_catalog,
):
    right_store, sparse_right_catalog = materialized_sparse_right_catalog
    missing_pixel = SPARSE_RIGHT_PIXELS[0]
    missing_partition = paths.pixel_catalog_file(
        sparse_right_catalog.hc_structure.catalog_base_dir, missing_pixel
    )
    assert missing_partition.is_file()
    missing_partition.unlink()

    # Reopening succeeds because HATS metadata still declares the deleted partition.
    missing_right_partition = lsdb.open_catalog(right_store)
    assert missing_pixel in missing_right_partition.get_healpix_pixels()
    assert not missing_partition.exists()
    nested_left = small_sky_order1_catalog.join_nested(
        missing_right_partition,
        left_on="id",
        right_on="object_id",
        nested_column_name="sources",
        how="left",
    )

    # Scientific sparsity is represented by nulls, while a missing object declared by
    # the HATS partition metadata is a storage error and must fail loudly.
    with pytest.raises(FileNotFoundError, match=re.escape(missing_partition.name)):
        nested_left.compute(progress_bar=False)


def test_join_nested_invalid_how(small_sky_order1_catalog, small_sky_order1_source_with_margin):
    with pytest.raises(ValueError, match="how"):
        small_sky_order1_catalog.join_nested(
            small_sky_order1_source_with_margin,
            left_on="id",
            right_on="object_id",
            nested_column_name="sources",
            how="right",
        )


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
def test_merge_asof(small_sky_catalog, small_sky_xmatch_catalog, direction):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.merge_asof(small_sky_xmatch_catalog, direction=direction, suffixes=suffixes)
    assert isinstance(joined.meta, npd.NestedFrame)
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


def test_merge_asof_overlapping_suffix(small_sky_catalog, small_sky_xmatch_catalog, helpers):
    suffixes = ("_a", "_b")
    joined = small_sky_catalog.merge_asof(
        small_sky_xmatch_catalog, direction="backward", suffixes=suffixes, suffix_method="overlapping_columns"
    )

    expected_columns = [
        "id_a",
        "ra_a",
        "dec_a",
        "ra_error_a",
        "dec_error_a",
        "id_b",
        "ra_b",
        "dec_b",
        "ra_error_b",
        "dec_error_b",
        "calculated_dist",
    ]
    assert np.all(joined.columns == expected_columns)

    joined_compute = joined.compute()

    assert np.all(joined_compute.columns == expected_columns)
    helpers.assert_schema_correct(joined)


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
    map_catalog = lsdb.read_hats(test_data_dir / "square_map")
    merge_lazy = small_sky_catalog.merge_map(map_catalog, merging_function, unused_kwarg="ignored")
    merge_result = merge_lazy.compute()
    assert len(merge_result) == small_sky_catalog.hc_structure.catalog_info.total_rows
