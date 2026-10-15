from typing import Any, cast

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from dask.delayed import delayed
from hats.io.paths import pixel_catalog_file
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath

import lsdb
from lsdb.streams import CatalogStream, InfiniteStream, catalog_streams
from lsdb.streams.catalog_streams import CrossMatchStream, _get_compression_ratio


def test_catalog_stream():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)

    # Test default iteration
    cat_stream = CatalogStream(catalog=cat)

    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk
    assert len(cat_iter) == 11

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test chunk size>1
    cat_stream = CatalogStream(catalog=cat, seed=1, partitions_per_chunk=2)
    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test shuffing=False
    cat_stream = CatalogStream(catalog=cat, seed=1, shuffle=False)
    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk
    assert len(cat_iter) == 11

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)


def test_infinite_stream():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    # Test infinite looping
    cat_stream = InfiniteStream(catalog=cat, seed=1)
    cat_iter = iter(cat_stream)

    # Check that we can sample beyond the number of partitions without error
    for _ in range(cat.npartitions * 2):
        next(cat_iter)

    # Check that length raises an error for InfiniteStream
    with pytest.raises(TypeError, match="Length is not defined for an InfiniteStream."):
        len(cat_iter)


def test_invalid_catalog_input():
    with pytest.raises(
        ValueError, match="The provided catalog input type <class 'str'> is not a lsdb.Catalog object."
    ):
        CatalogStream(catalog=cast(lsdb.Catalog, "not a catalog"))


def test_rng_split():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    cat_stream = CatalogStream(catalog=cat, seed=1)
    cat_iter = iter(cat_stream)

    # Check that the RNG is properly split and produces different sequences for different iterations
    first_partitions = cat_iter.partitions_left.copy()
    next(cat_iter)  # Advance the iterator to change the RNG state
    second_partitions = cat_iter.partitions_left.copy()

    assert not np.array_equal(
        first_partitions, second_partitions
    ), "RNG should produce different sequences after splitting."


def test_stream_from_search_filter():
    """test to make sure pixel set handoff from a search filter works"""
    # explicit test for https://github.com/astronomy-commons/lsdb/issues/1549

    cat = lsdb.generate_catalog(1000, 2, seed=1, ra_range=(0.0, 10.0), dec_range=(0, 10.0), partition_rows=20)
    cat = cat.cone_search(ra=0, dec=0, radius_arcsec=1000)

    # If this runs without error, then the stream is working
    for i, chunk in enumerate(CatalogStream(cat)):
        if i > 0:
            break
        print(chunk)


def _crossmatch_stream_catalogs(monkeypatch):
    """Two small overlapping catalogs for CrossMatchStream tests.

    CrossMatchStream reads each catalog's on-disk point-map skymap to
    estimate match fractions; in-memory catalogs have none, so serve a
    synthetic all-ones skymap (every pixel estimated to match fully)."""
    monkeypatch.setattr(
        "lsdb.streams.catalog_streams.read_skymap",
        lambda hc_catalog, path: np.ones(hp.order2npix(5)),
    )
    left_df = pd.DataFrame(
        {
            "ra": np.linspace(15.0, 25.0, 64),
            "dec": np.linspace(34.0, 44.0, 64),
            "id": np.arange(64),
        }
    )
    right_df = pd.DataFrame(
        {
            "ra": np.linspace(15.0, 25.0, 32),
            "dec": np.linspace(34.0, 44.0, 32),
            "id": np.arange(32),
        }
    )
    left = lsdb.from_dataframe(
        left_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=4,
        highest_order=5,
        margin_threshold=30,
    )
    right = lsdb.from_dataframe(
        right_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=4,
        highest_order=5,
        margin_threshold=30,
    )
    return left, right


def test_crossmatch_stream_yields_left_joined_rows(monkeypatch):
    left, right = _crossmatch_stream_catalogs(monkeypatch)
    stream = CrossMatchStream(
        left,
        {"other": right},
        client=None,
        partitions_per_chunk=2,
        seed=1,
        count_fraction_threshold=0.0,
    )
    cat_iter = iter(stream)
    chunk = next(cat_iter)
    assert len(chunk) > 0
    # never-skip threshold: every row keeps the right catalog's columns,
    # suffixed with the right catalog's name
    assert f"id_{right.name}" in chunk.columns
    assert len(chunk["id"]) == chunk["id"].notna().sum()


def test_crossmatch_stream_draws_every_aligned_pixel(monkeypatch):
    """A finer right catalog splits anchor partitions; every aligned pixel is drawn."""
    monkeypatch.setattr(
        "lsdb.streams.catalog_streams.read_skymap",
        lambda hc_catalog, path: np.ones(hp.order2npix(5)),
    )
    ras, decs = np.meshgrid(np.linspace(15.0, 25.0, 20), np.linspace(34.0, 44.0, 20))
    df = pd.DataFrame({"ra": ras.ravel(), "dec": decs.ravel(), "id": np.arange(ras.size)})
    left = lsdb.from_dataframe(df, lowest_order=3, highest_order=3, margin_threshold=30)
    right = lsdb.from_dataframe(df, lowest_order=3, highest_order=5, partition_rows=25, margin_threshold=30)
    stream = CrossMatchStream(left, {"other": right}, seed=1, count_fraction_threshold=0.0)
    n_pixels = len(stream._pixels)
    assert n_pixels > left.npartitions

    cat_iter = iter(stream)
    assert sorted(cat_iter.partitions_left) == list(range(n_pixels))


def test_stream_submits_next_chunk_before_waiting_for_result():
    """The next chunk is submitted before the current result is awaited.

    Graph construction and scheduling for chunk N+1 must overlap chunk N's
    computation (the class docstring's pre-fetch claim), not run on the
    consumer's critical path after it. Ordering is observable by logging
    submissions and result reads; with a real dask client this is what
    hides CrossMatchStream's per-pixel graph construction behind fetches."""
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    events = []

    class _LoggingFuture:  # pylint: disable=too-few-public-methods
        def __init__(self, inner):
            self.inner = inner

        def result(self):
            value = self.inner.result()
            events.append("read")
            return value

    class RecordingStream(CatalogStream):
        def submit_next_partitions(self, partitions) -> Any:
            future = super().submit_next_partitions(partitions)
            events.append(f"submit:{len(partitions)}")
            return _LoggingFuture(future)

    stream = RecordingStream(catalog=cat, seed=1, partitions_per_chunk=2)
    cat_iter = iter(stream)
    assert events == ["submit:2"]  # the first chunk is pre-submitted

    next(cat_iter)
    # the second chunk was submitted BEFORE the first result was read
    assert events == ["submit:2", "submit:2", "read"]

    # iteration still yields every row exactly once
    total = 0
    for chunk in RecordingStream(catalog=cat, seed=1, partitions_per_chunk=2):
        total += len(chunk)
    assert total == len(cat)


def test_compression_ratio_weights_columns_row_groups_and_source_pixels(
    small_sky_order1_catalog, tmp_path, monkeypatch
):
    catalog = small_sky_order1_catalog
    compressed_bytes = uncompressed_bytes = 0
    chunk_ratios = []
    for i, pixel in enumerate(catalog.get_healpix_pixels()):
        size = 128 * (i + 1)
        table = pa.table(
            {
                "flux": np.zeros(size),
                "spectrum": np.random.default_rng(i).normal(size=size),
                "ignored": np.arange(size),
            }
        )
        path = tmp_path / f"{pixel.pixel}.parquet"
        pq.write_table(
            table, path, compression="gzip", use_dictionary=False, row_group_size=100, write_statistics=False
        )
        metadata = pq.read_metadata(path)
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in (0, 1):
                chunk = row_group.column(column_index)
                compressed_bytes += chunk.total_compressed_size
                uncompressed_bytes += chunk.total_uncompressed_size
                chunk_ratios.append(chunk.total_compressed_size / chunk.total_uncompressed_size)

    monkeypatch.setattr(
        catalog.loading_config,
        "path_generator",
        lambda base, pixel, query, suffix: UPath(tmp_path / f"{pixel.pixel}.parquet"),
    )

    def no_data_read(*_args, **_kwargs):
        pytest.fail("Scoring must not read science values or compute catalog data")

    monkeypatch.setattr(pq, "read_table", no_data_read)
    monkeypatch.setattr(lsdb.Catalog, "compute", no_data_read)
    expected = compressed_bytes / uncompressed_bytes
    assert expected != pytest.approx(np.mean(chunk_ratios))
    assert _get_compression_ratio(catalog, HealpixPixel(0, 11), ["flux", "spectrum"]) == expected


@pytest.mark.parametrize("catalog_name", ["small_sky", "small_sky_npix_alt_suffix", "small_sky_npix_as_dir"])
def test_compression_ratio_finer_pixel_and_file_layouts(test_data_dir, catalog_name):
    catalog = lsdb.open_catalog(test_data_dir / catalog_name)
    parent = _get_compression_ratio(catalog, HealpixPixel(0, 11), ["ra_error", "dec_error"])
    assert parent > 0
    assert _get_compression_ratio(catalog, HealpixPixel(1, 44), ["ra_error", "dec_error"]) == parent
    assert _get_compression_ratio(catalog, HealpixPixel(0, 0), ["ra_error"]) == 0


def test_compression_ratio_nested_features(small_sky_with_nested_sources):
    catalog = small_sky_with_nested_sources
    pixel = catalog.get_healpix_pixels()[0]
    metadata = pq.read_metadata(pixel_catalog_file(catalog.hc_structure.catalog_base_dir, pixel))
    chunks = [
        metadata.row_group(i).column(j)
        for i in range(metadata.num_row_groups)
        for j in range(metadata.num_columns)
    ]
    chunks = [chunk for chunk in chunks if chunk.path_in_schema.startswith("sources.")]
    expected = sum(chunk.total_compressed_size for chunk in chunks) / sum(
        chunk.total_uncompressed_size for chunk in chunks
    )
    assert _get_compression_ratio(catalog, pixel, ["sources"]) == expected
    subcolumns = catalog.meta.get_subcolumns()
    assert _get_compression_ratio(catalog, pixel, subcolumns) == expected
    # Selecting a group and its subcolumns must not count chunks twice.
    assert _get_compression_ratio(catalog, pixel, ["sources", *subcolumns]) == expected


def test_compression_ratio_missing_columns(small_sky_catalog):
    with pytest.raises(ValueError, match="missing.*missing from footer"):
        _get_compression_ratio(small_sky_catalog, HealpixPixel(0, 11), ["ra_error", "missing"])


@pytest.mark.parametrize(
    "compression_options",
    [
        {"compression_columns": ["id"]},
        {"compression_ratio_threshold": 0.5},
        {"compression_columns": [], "compression_ratio_threshold": 0.5},
        {"compression_columns": "id", "compression_ratio_threshold": 0.5},
        {"compression_columns": [1], "compression_ratio_threshold": 0.5},
        {"compression_columns": [""], "compression_ratio_threshold": 0.5},
        {"compression_columns": ["id"], "compression_ratio_threshold": -1},
        {"compression_columns": ["id"], "compression_ratio_threshold": np.nan},
        {"compression_columns": ["id"], "compression_ratio_threshold": np.inf},
        {"compression_columns": ["id"], "compression_ratio_threshold": "0.5"},
    ],
)
def test_crossmatch_stream_invalid_compression_options(monkeypatch, compression_options):
    left, right = _crossmatch_stream_catalogs(monkeypatch)
    with pytest.raises(ValueError, match="Specify both"):
        CrossMatchStream(left, {"other": right, **compression_options}, count_fraction_threshold=0)


def test_crossmatch_stream_compression_requires_disk_catalog(monkeypatch):
    left, right = _crossmatch_stream_catalogs(monkeypatch)
    with pytest.raises(ValueError, match="on-disk right catalog"):
        CrossMatchStream(
            left,
            {"other": right, "compression_columns": ["id"], "compression_ratio_threshold": 0.5},
            count_fraction_threshold=0,
        )


@pytest.mark.parametrize(
    "threshold, coverage, keep", [(0, 0, True), (10, 0, False), (0, 2, False), (None, 0, True)]
)
def test_crossmatch_stream_compression_filter(
    small_sky_catalog,
    small_sky_order1_catalog,
    small_sky_order1_margin_1deg_catalog,
    monkeypatch,
    threshold,
    coverage,
    keep,
):
    right = small_sky_order1_catalog
    right.margin = small_sky_order1_margin_1deg_catalog
    kwargs = {"other": right}
    if threshold is not None:
        kwargs.update(compression_columns=["ra_error", "dec_error"], compression_ratio_threshold=threshold)
    original_kwargs = kwargs.copy()
    stream = CrossMatchStream(small_sky_catalog, kwargs, count_fraction_threshold=coverage)
    assert kwargs == original_kwargs
    if coverage > 1 or threshold is None:
        monkeypatch.setattr(
            catalog_streams,
            "_get_compression_ratio",
            lambda *args: pytest.fail("Disabled/coverage-rejected filters must not read footers"),
        )
    chunk = stream.submit_next_partitions(np.arange(len(stream._pixels))).result()
    assert len(chunk) == len(small_sky_catalog)
    assert chunk["id"].notna().all()
    assert chunk[f"id_{right.name}"].notna().sum() == (len(chunk) if keep else 0)


def test_crossmatch_stream_per_catalog_thresholds_are_inclusive(
    small_sky_catalog,
    small_sky_order1_catalog,
    small_sky_order1_margin_1deg_catalog,
    small_sky_xmatch_with_margin,
    monkeypatch,
):
    right = small_sky_order1_catalog
    right.margin = small_sky_order1_margin_1deg_catalog
    monkeypatch.setattr(catalog_streams, "_get_compression_ratio", lambda *args: 0.5)
    stream = CrossMatchStream(
        small_sky_catalog,
        {"other": right, "compression_columns": ["ra_error"], "compression_ratio_threshold": 0.5},
        {
            "other": small_sky_xmatch_with_margin,
            "compression_columns": ["ra"],
            "compression_ratio_threshold": 0.51,
        },
        count_fraction_threshold=0,
    )
    chunk = stream.submit_next_partitions(np.arange(len(stream._pixels))).result()
    assert len(chunk) == len(small_sky_catalog)
    assert chunk[f"id_{right.name}"].notna().all()
    assert chunk[f"id_{small_sky_xmatch_with_margin.name}"].isna().all()


def test_compression_ratio_empty_metadata(small_sky_catalog, tmp_path, monkeypatch):
    path = UPath(tmp_path / "empty.parquet")
    pq.write_metadata(pa.schema([("ra_error", pa.int64())]), path.path)
    monkeypatch.setattr(small_sky_catalog.loading_config, "path_generator", lambda *args: path)
    with pytest.raises(ValueError, match="footer metadata"):
        _get_compression_ratio(small_sky_catalog, HealpixPixel(0, 11), ["ra_error"])


STAR_RADIUS = 10.0


def _star_catalog(rows, name, order):
    df = pd.DataFrame(rows, columns=["id", "ra", "dec"])
    return lsdb.from_dataframe(
        df, catalog_name=name, lowest_order=order, highest_order=order, margin_threshold=60
    )


def _star_catalogs(monkeypatch, anchor_order=2, right_order=2):
    """An anchor and two right catalogs covering every star-mode case.

    Anchor row 1 matches both right catalogs, row 2 only C, row 3 neither. B and C
    each have a row unmatched inside the anchor footprint (12, 23) and one outside
    it (13, 24); B's row 14 and C's row 23 sit at the same position, with no anchor.
    """
    monkeypatch.setattr(
        "lsdb.streams.catalog_streams.read_skymap",
        lambda hc_catalog, path: np.ones(hp.order2npix(5)),
    )
    anchor = _star_catalog([(1, 10.0, 10.0), (2, 10.5, 10.0), (3, 11.0, 10.0)], "A", anchor_order)
    right_b = _star_catalog(
        [(11, 10.0005, 10.0), (12, 10.3, 10.0), (13, 120.0, -30.0), (14, 10.8, 10.0)], "B", right_order
    )
    right_c = _star_catalog(
        [(21, 10.5005, 10.0), (22, 10.0007, 10.0), (23, 10.8, 10.0), (24, 200.0, 40.0)], "C", right_order
    )
    return anchor, right_b, right_c


def _stream_all_pixels(stream):
    return stream.submit_next_partitions(np.arange(stream.npartitions)).result()


def _id_rows(frame):
    return sorted(frame[["id", "id_B", "id_C"]].astype("Int64").fillna(-1).values.tolist())


@pytest.mark.parametrize(("anchor_order", "right_order"), [(2, 2), (4, 1), (1, 4)])
def test_crossmatch_stream_star_outer(monkeypatch, anchor_order, right_order):
    anchor, right_b, right_c = _star_catalogs(monkeypatch, anchor_order, right_order)
    stream = CrossMatchStream(
        anchor,
        {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
        {"other": right_c, "how": "outer", "radius_arcsec": STAR_RADIUS},
        seed=1,
        count_fraction_threshold=0.0,
    )
    chunk = _stream_all_pixels(stream)

    assert _id_rows(chunk) == [
        [-1, -1, 23],
        [-1, 12, -1],
        [-1, 14, -1],
        [1, 11, 22],
        [2, -1, 21],
        [3, -1, -1],
    ]
    assert list(chunk.columns) == list(stream._star.assembly.meta.columns)
    assert chunk.dtypes.equals(stream._star.assembly.meta.dtypes)
    anchored = chunk["id"].notna()
    assert chunk.loc[anchored, "_ra"].tolist() == chunk.loc[anchored, "ra"].tolist()
    b_only = chunk["id_B"].notna() & ~anchored
    assert chunk.loc[b_only, "_ra"].tolist() == chunk.loc[b_only, "ra_B"].tolist()


def test_crossmatch_stream_star_left_right_catalog_adds_no_rows(monkeypatch):
    anchor, right_b, right_c = _star_catalogs(monkeypatch)
    stream = CrossMatchStream(
        anchor,
        {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
        {"other": right_c, "radius_arcsec": STAR_RADIUS},
        seed=1,
        count_fraction_threshold=0.0,
    )

    assert _id_rows(_stream_all_pixels(stream)) == [
        [-1, 12, -1],
        [-1, 14, -1],
        [1, 11, 22],
        [2, -1, 21],
        [3, -1, -1],
    ]


def test_crossmatch_stream_star_skipped_right_catalog(monkeypatch):
    anchor, right_b, right_c = _star_catalogs(monkeypatch)
    stream = CrossMatchStream(
        anchor,
        {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
        {"other": right_c, "how": "outer", "radius_arcsec": STAR_RADIUS},
        seed=1,
        count_fraction_threshold=0.0,
    )
    monkeypatch.setattr(stream, "_right_catalog_mask", lambda pixel: [True, False])

    assert _id_rows(_stream_all_pixels(stream)) == [
        [-1, 12, -1],
        [-1, 14, -1],
        [1, 11, -1],
        [2, -1, -1],
        [3, -1, -1],
    ]


@pytest.mark.parametrize(
    ("anchor_order", "right_orders"), [(4, (1, 1)), (1, (4, 4)), (3, (3, 3)), (2, (1, 4))]
)
def test_crossmatch_stream_star_matches_pairwise_crossmatches(monkeypatch, anchor_order, right_orders):
    """Across many partitions, every anchor row and unmatched right row is emitted once."""
    monkeypatch.setattr(
        "lsdb.streams.catalog_streams.read_skymap",
        lambda hc_catalog, path: np.ones(hp.order2npix(5)),
    )
    rng = np.random.default_rng(0)
    anchor_ra, anchor_dec = rng.uniform(10, 30, 300), rng.uniform(-10, 10, 300)
    anchor = _star_catalog(zip(range(300), anchor_ra, anchor_dec), "A", anchor_order)
    rights = []
    for name, first_id, right_order in (("B", 1000, right_orders[0]), ("C", 2000, right_orders[1])):
        matched = rng.choice(300, 100, replace=False)
        ras = np.concatenate([anchor_ra[matched] + 0.001, rng.uniform(0, 40, 200)])
        decs = np.concatenate([anchor_dec[matched], rng.uniform(-20, 20, 200)])
        rights.append(_star_catalog(zip(range(first_id, first_id + 300), ras, decs), name, right_order))
    stream = CrossMatchStream(
        anchor,
        *({"other": right, "how": "outer", "radius_arcsec": STAR_RADIUS} for right in rights),
        seed=1,
        count_fraction_threshold=0.0,
    )
    chunk = pd.DataFrame(_stream_all_pixels(stream))

    anchored = chunk[chunk["id"].notna()]
    assert sorted(anchored["id"]) == list(range(300))
    for right in rights:
        pairwise = pd.DataFrame(
            anchor.crossmatch(
                right,
                how="outer",
                outer_footprint="left",
                radius_arcsec=STAR_RADIUS,
                suffixes=("", f"_{right.name}"),
                suffix_method="all_columns",
            ).compute()
        )
        id_column = f"id_{right.name}"
        expected = pairwise[pairwise["id"].notna()].set_index("id")[id_column].sort_index()
        assert anchored.set_index("id")[id_column].sort_index().equals(expected)
        right_only = chunk[chunk[id_column].notna() & chunk["id"].isna()][id_column]
        expected_only = pairwise[pairwise["id"].isna()][id_column]
        assert len(expected_only) > 0
        assert sorted(right_only) == sorted(expected_only)


def test_crossmatch_stream_star_reads_anchor_once_per_pixel(monkeypatch):
    anchor, right_b, right_c = _star_catalogs(monkeypatch)
    stream = CrossMatchStream(
        anchor,
        {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
        {"other": right_c, "how": "outer", "radius_arcsec": STAR_RADIUS},
        seed=1,
        count_fraction_threshold=0.0,
    )
    pixel = stream._pixels[0]
    right_parts = [catalog_streams._partition_delayeds(result, pixel) for result in stream._star.results]
    task = delayed(list)(right_parts)
    anchor_key = stream._star.anchor._operation.key_name

    anchor_tasks = [key for key in dict(task.__dask_graph__()) if key[0] == anchor_key]
    assert len(anchor_tasks) == 1


def test_crossmatch_stream_rejects_unsupported_how(monkeypatch):
    left, right = _crossmatch_stream_catalogs(monkeypatch)
    with pytest.raises(ValueError, match="how='left' or how='outer'"):
        CrossMatchStream(left, {"other": right, "how": "inner"}, count_fraction_threshold=0.0)


def test_crossmatch_stream_star_outer_requires_anchor_margin(monkeypatch):
    anchor, right_b, _ = _star_catalogs(monkeypatch)
    anchor.margin = None
    with pytest.raises(ValueError, match="left catalog to have a margin cache"):
        CrossMatchStream(
            anchor,
            {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
            count_fraction_threshold=0.0,
        )
