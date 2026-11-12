from typing import cast

import astropy.units as u
import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pytest
from dask.delayed import delayed
from hats.pixel_math.healpix_pixel import HealpixPixel
from mocpy import MOC

import lsdb
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.streams import CatalogStream, InfiniteStream, catalog_streams
from lsdb.streams.catalog_streams import CrossMatchStream


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


def test_stream_submits_next_chunk_before_waiting_for_result():
    """Chunk N+1 is submitted before chunk N's result is read, so their work overlaps."""
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    events = []

    class _LoggingFuture:  # pylint: disable=too-few-public-methods
        def __init__(self, inner):
            self.inner = inner

        def result(self):
            events.append("read")
            return self.inner.result()

    class RecordingStream(CatalogStream):
        def submit_next_partitions(self, partitions):
            events.append("submit")
            return _LoggingFuture(super().submit_next_partitions(partitions))

    cat_iter = iter(RecordingStream(catalog=cat, seed=1, partitions_per_chunk=2))
    assert events == ["submit"]  # the first chunk is pre-submitted
    next(cat_iter)
    assert events == ["submit", "submit", "read"]


def test_crossmatch_stream_draws_every_aligned_pixel():
    """A finer right catalog splits anchor partitions; every aligned pixel is drawn."""
    ras, decs = np.meshgrid(np.linspace(15.0, 25.0, 20), np.linspace(34.0, 44.0, 20))
    df = pd.DataFrame({"ra": ras.ravel(), "dec": decs.ravel(), "id": np.arange(ras.size)})
    left = lsdb.from_dataframe(df, lowest_order=3, highest_order=3, margin_threshold=30)
    right = lsdb.from_dataframe(df, lowest_order=3, highest_order=5, partition_rows=25, margin_threshold=30)
    stream = CrossMatchStream(left, {"other": right}, seed=1)
    n_pixels = len(stream._pixels)
    assert n_pixels > left.npartitions

    cat_iter = iter(stream)
    assert sorted(cat_iter.partitions_left) == list(range(n_pixels))


STAR_RADIUS = 10.0


def _star_catalog(rows, name, order):
    df = pd.DataFrame(rows, columns=["id", "ra", "dec"])
    return lsdb.from_dataframe(
        df, catalog_name=name, lowest_order=order, highest_order=order, margin_threshold=60
    )


def _star_catalogs(anchor_order=2, right_order=2):
    """An anchor and two right catalogs covering every star-mode case.

    Anchor row 1 matches both right catalogs, row 2 only C, row 3 neither. B and C
    each have a row unmatched inside the anchor's pixels (12, 23) and one outside
    them (13, 24); B's row 14 and C's row 23 sit at the same position, with no anchor.
    Rows 12, 14 and 23 lie outside the anchor's order-10 data MOC. Anchor row 4 and
    B's row 15 straddle an order-2 pixel boundary, so they match across it.
    """
    anchor = _star_catalog(
        [(1, 10.0, 10.0), (2, 10.5, 10.0), (3, 11.0, 10.0), (4, 0.0, 0.0)], "A", anchor_order
    )
    right_b = _star_catalog(
        [(11, 10.0005, 10.0), (12, 10.3, 10.0), (13, 120.0, -30.0), (14, 10.8, 10.0), (15, 0.001, 0.0)],
        "B",
        right_order,
    )
    right_c = _star_catalog(
        [(21, 10.5005, 10.0), (22, 10.0007, 10.0), (23, 10.8, 10.0), (24, 200.0, 40.0)], "C", right_order
    )
    return anchor, right_b, right_c


def _stream_all_pixels(stream):
    return stream.submit_next_partitions(np.arange(stream.npartitions)).result()


def _id_rows(frame, columns=("id", "id_B", "id_C")):
    return sorted(frame[list(columns)].astype("Int64").fillna(-1).values.tolist())


def _in_pixels(ra, dec, pixels):
    """Whether each coordinate falls in one of ``pixels``, from HEALPix directly."""
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    inside = np.zeros(len(ra), dtype=bool)
    for order in {pixel.order for pixel in pixels}:
        indices = [pixel.pixel for pixel in pixels if pixel.order == order]
        inside |= np.isin(hp.radec2pix(order, ra, dec), indices)
    return inside


def _pixel_tree(catalog):
    return catalog.hc_structure.pixel_tree.to_moc()


ANCHOR_ROWS = [[1, 11, 22], [2, -1, 21], [3, -1, -1], [4, 15, -1]]
UNMATCHED_IN_PIXELS = [[-1, -1, 23], [-1, 12, -1], [-1, 14, -1]]
CONE = MOC.from_cone(lon=10.0 * u.deg, lat=10.0 * u.deg, radius=0.35 * u.deg, max_depth=12)


@pytest.mark.parametrize(
    ("orders", "hows", "footprint", "expected"),
    [
        # By default only the anchor's data MOC is kept, which excludes 12, 14 and 23
        *(((orders, ("outer", "outer"), None, ANCHOR_ROWS)) for orders in ((2, 2), (4, 1), (1, 4))),
        *(
            ((orders, ("outer", "outer"), "pixel_tree", UNMATCHED_IN_PIXELS + ANCHOR_ROWS))
            for orders in ((2, 2), (4, 1), (1, 4))
        ),
        # A given footprint is clipped to the anchor's pixels, and also restricts the anchor
        ((2, 2), ("outer", "outer"), MOC.from_string("0/0-11"), UNMATCHED_IN_PIXELS + ANCHOR_ROWS),
        ((2, 2), ("outer", "outer"), CONE, [[-1, 12, -1], [1, 11, 22]]),
        # A left right catalog adds no rows
        ((2, 2), ("outer", "left"), "pixel_tree", [[-1, 12, -1], [-1, 14, -1], *ANCHOR_ROWS]),
    ],
)
def test_crossmatch_stream_star_rows(orders, hows, footprint, expected):
    anchor, right_b, right_c = _star_catalogs(*orders)
    stream = CrossMatchStream(
        anchor,
        *(
            {"other": right, "how": how, "radius_arcsec": STAR_RADIUS}
            for right, how in zip((right_b, right_c), hows, strict=True)
        ),
        seed=1,
        footprint=_pixel_tree(anchor) if isinstance(footprint, str) else footprint,
    )
    chunk = _stream_all_pixels(stream)

    assert _id_rows(chunk) == sorted(expected)
    assert list(chunk.columns) == list(stream._assembly.meta.columns)
    assert chunk.dtypes.equals(stream._assembly.meta.dtypes)
    anchored, b_only = chunk["id"].notna(), chunk["id_B"].notna() & chunk["id"].isna()
    assert chunk.loc[anchored, "_ra"].tolist() == chunk.loc[anchored, "ra"].tolist()
    assert chunk.loc[b_only, "_ra"].tolist() == chunk.loc[b_only, "ra_B"].tolist()


@pytest.mark.parametrize("n_neighbors", [None, 1])
def test_crossmatch_stream_left_join(n_neighbors):
    """Left-only streams keep anchor pixels no right catalog covers, and add no _ra/_dec."""
    anchor = _star_catalog([(1, 10.0, 10.0), (2, 200.0, 40.0)], "A", 2)
    right = _star_catalog([(11, 10.0005, 10.0)], "B", 2)
    stream = CrossMatchStream(
        anchor,
        {"other": right, "radius_arcsec": STAR_RADIUS, "n_neighbors": n_neighbors},
        partitions_per_chunk=2,
        seed=1,
    )
    chunk = next(iter(stream))

    assert _id_rows(chunk, columns=("id", "id_B")) == [[1, 11], [2, -1]]
    assert list(chunk.columns) == list(anchor.columns) + [f"{column}_B" for column in right.columns]
    assert chunk.index.name == anchor.meta.index.name


@pytest.mark.parametrize(("how", "pixel_tree"), [("left", False), ("outer", False), ("outer", True)])
@pytest.mark.parametrize(("anchor_order", "right_orders"), [(4, (1, 1)), (1, (4, 4)), (2, (1, 4))])
def test_crossmatch_stream_star_matches_pairwise_crossmatches(anchor_order, right_orders, how, pixel_tree):
    """Across many partitions, every anchor row and unmatched right row is emitted once."""
    rng = np.random.default_rng(0)
    anchor_ra, anchor_dec = rng.uniform(10, 30, 300), rng.uniform(-10, 10, 300)
    anchor = _star_catalog(zip(range(300), anchor_ra, anchor_dec), "A", anchor_order)
    rights = []
    for name, first_id, right_order in (("B", 1000, right_orders[0]), ("C", 2000, right_orders[1])):
        matched = rng.choice(300, 100, replace=False)
        # 36" from an anchor row: unmatched, but often inside the anchor's data MOC
        near_miss = rng.choice(300, 100, replace=False)
        ras = np.concatenate(
            [anchor_ra[matched] + 0.001, anchor_ra[near_miss] + 0.01, rng.uniform(0, 40, 200)]
        )
        decs = np.concatenate([anchor_dec[matched], anchor_dec[near_miss], rng.uniform(-20, 20, 200)])
        rights.append(_star_catalog(zip(range(first_id, first_id + 400), ras, decs), name, right_order))
    stream = CrossMatchStream(
        anchor,
        *({"other": right, "how": how, "radius_arcsec": STAR_RADIUS} for right in rights),
        seed=1,
        footprint=_pixel_tree(anchor) if pixel_tree else None,
    )
    chunk = pd.DataFrame(_stream_all_pixels(stream))

    anchored = chunk[chunk["id"].notna()]
    assert sorted(anchored["id"]) == list(range(300))
    for right in rights:
        # The pairwise crossmatch. An outer one is restricted to the anchor's pixels by
        # HEALPix directly, or by default to the anchor's data MOC.
        pairwise = pd.DataFrame(
            anchor.crossmatch(
                right,
                how=how,
                radius_arcsec=STAR_RADIUS,
                suffixes=("", f"_{right.name}"),
                suffix_method="all_columns",
            ).compute()
        )
        if how == "outer":
            if pixel_tree:
                inside = _in_pixels(pairwise["_ra"], pairwise["_dec"], anchor.get_healpix_pixels())
            else:
                inside = anchor.hc_structure.moc.contains_lonlat(
                    pairwise["_ra"].to_numpy(dtype=float) * u.deg,
                    pairwise["_dec"].to_numpy(dtype=float) * u.deg,
                )
            assert not inside.all()
            pairwise = pairwise[inside]
        id_column = f"id_{right.name}"
        expected = pairwise[pairwise["id"].notna()].set_index("id")[id_column].sort_index()
        assert anchored.set_index("id")[id_column].sort_index().equals(expected)
        right_only = chunk[chunk[id_column].notna() & chunk["id"].isna()][id_column]
        expected_only = pairwise[pairwise["id"].isna()][id_column]
        assert (len(expected_only) > 0) == (how == "outer")
        assert sorted(right_only) == sorted(expected_only)


def test_crossmatch_stream_star_reads_anchor_once_per_pixel():
    anchor, right_b, right_c = _star_catalogs()
    stream = CrossMatchStream(
        anchor,
        {"other": right_b, "how": "outer", "radius_arcsec": STAR_RADIUS},
        {"other": right_c, "how": "outer", "radius_arcsec": STAR_RADIUS},
        seed=1,
    )
    pixel = stream._pixels[0]
    right_parts = [catalog_streams._partition_delayeds(result, pixel) for result in stream._results]
    task = delayed(list)(right_parts)
    anchor_key = stream._anchor._operation.key_name

    anchor_tasks = [
        key for key in dict(task.__dask_graph__()) if isinstance(key, tuple) and key[0] == anchor_key
    ]
    assert len(anchor_tasks) == 1


def test_crossmatch_stream_star_never_reads_right_only_sky():
    anchor, right_b, right_c = _star_catalogs()
    star_kwargs = [
        {"other": right, "how": "outer", "radius_arcsec": STAR_RADIUS} for right in (right_b, right_c)
    ]
    stream = CrossMatchStream(anchor, *star_kwargs, seed=1)
    parts = [
        catalog_streams._partition_delayeds(result, pixel)
        for pixel in stream._pixels
        for result in stream._results
    ]
    stream_graph = set(dict(delayed(list)(parts).__dask_graph__()))

    for right, (ra, dec), kwargs in zip(
        (right_b, right_c), ((120.0, -30.0), (200.0, 40.0)), stream.crossmatch_kwargs, strict=True
    ):
        right_only_pixel = HealpixPixel(2, hp.radec2pix(2, ra, dec)[0].item())
        assert right_only_pixel in right.get_healpix_pixels()
        assert right_only_pixel not in anchor.get_healpix_pixels()
        right_only_keys = set(right._operation.build([right_only_pixel]).graph)
        assert not right_only_keys & stream_graph
        # Without the footprint filter the outer crossmatch would read it
        union = anchor.crossmatch(**kwargs)
        assert right_only_keys & set(union._operation.build(union.get_healpix_pixels()).graph)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"how": "inner"}, "how='left' or how='outer'"),
        ({"n_neighbors": 2}, "CrossMatchStream supports at most one neighbor per anchor row"),
        ({"how": "outer", "radius_arcsec": STAR_RADIUS}, "left catalog to have a margin cache"),
    ],
)
def test_crossmatch_stream_rejects_bad_arguments(options, match):
    anchor, right, _ = _star_catalogs()
    anchor.margin = None  # only reached once the other arguments are valid
    with pytest.raises(ValueError, match=match):
        CrossMatchStream(anchor, {"other": right, **options})


@pytest.mark.parametrize("how", ["left", "outer"])
def test_crossmatch_stream_rejects_multiple_algorithm_matches(how):
    """Algorithm-specified neighbors must not multiply rows across right catalogs."""
    anchor = _star_catalog([(1, 10.0, 10.0)], "A", 2)
    rights = [_star_catalog([(11, 10.0005, 10.0), (12, 10.0007, 10.0)], name, 2) for name in ("B", "C")]
    stream = CrossMatchStream(
        anchor,
        *(
            {
                "other": right,
                "how": how,
                "algorithm": KdTreeCrossmatch(n_neighbors=2, radius_arcsec=STAR_RADIUS),
            }
            for right in rights
        ),
    )
    with pytest.raises(pd.errors.MergeError, match="not a many-to-one merge"):
        _stream_all_pixels(stream)


def test_crossmatch_stream_right_suffixes():
    """The same right catalog twice needs distinct suffixes, which the stream keeps."""
    anchor, right, _ = _star_catalogs()
    # Anchor rows 1 and 4 are 1.8" and 3.6" from B's rows 11 and 15
    near, far = {"other": right, "radius_arcsec": 2.0}, {"other": right, "radius_arcsec": STAR_RADIUS}
    with pytest.raises(ValueError, match="id_B.*distinct suffixes"):
        CrossMatchStream(anchor, near, far)

    stream = CrossMatchStream(
        anchor,
        {**near, "suffixes": ("", "_near")},
        {**far, "suffixes": ("_ignored", "_far")},
    )
    chunk = _stream_all_pixels(stream)
    assert _id_rows(chunk, ("id", "id_near", "id_far")) == [
        [1, 11, 11],
        [2, -1, -1],
        [3, -1, -1],
        [4, -1, 15],
    ]


@pytest.mark.parametrize(
    ("footprint", "min_catalogs", "expected"),
    [
        # B's 14 and C's 23 share a spot the anchor lacks, outside its data MOC
        (None, 2, [[1, 11, 22], [2, -1, 21]]),
        ("pixel_tree", 2, [[-1, -1, 23], [-1, 14, -1], [1, 11, 22], [2, -1, 21]]),
        (None, 3, [[1, 11, 22]]),
        ("pixel_tree", 3, [[1, 11, 22]]),
    ],
)
def test_crossmatch_stream_min_catalogs(footprint, min_catalogs, expected):
    """Anchor row 3 and B's 12 sit alone, and row 4 and B's 15 are split by a pixel border."""
    anchor, right_b, right_c = _star_catalogs()
    star_kwargs = [
        {"other": right, "how": "outer", "radius_arcsec": STAR_RADIUS} for right in (right_b, right_c)
    ]
    with pytest.raises(ValueError, match="min_catalogs"):
        CrossMatchStream(anchor, *star_kwargs, min_catalogs=4)

    stream = CrossMatchStream(
        anchor,
        *star_kwargs,
        footprint=_pixel_tree(anchor) if footprint else None,
        min_catalogs=min_catalogs,
    )
    assert _id_rows(_stream_all_pixels(stream)) == expected


def test_crossmatch_stream_min_catalogs_ignores_removed_partitions():
    rows = [(1, 10.0, 10.0), (2, 200.0, 40.0)]
    anchor = _star_catalog(rows, "A", 2)
    right = _star_catalog(rows, "B", 2)
    right = right.get_partition(2, hp.radec2pix(2, 200.0, 40.0)[0].item())

    stream = CrossMatchStream(anchor, {"other": right}, min_catalogs=2)

    assert _id_rows(_stream_all_pixels(stream), ("id", "id_B")) == [[2, 2]]


def test_min_catalogs_coverage_work_scales_with_catalogs(small_sky_catalog, monkeypatch):
    """Bound coverage operations rather than wall-clock time to catch combinatorial growth."""
    n_catalogs, min_catalogs = 20, 10
    expected = catalog_streams._data_coverage(small_sky_catalog)
    intersection = MOC.intersection
    intersections = 0

    def counted_intersection(moc, *others):
        nonlocal intersections
        intersections += 1
        assert intersections <= 2 * n_catalogs * min_catalogs
        return intersection(moc, *others)

    monkeypatch.setattr(MOC, "intersection", counted_intersection)
    _, footprint = catalog_streams._restrict_to_footprint(
        small_sky_catalog, None, [small_sky_catalog] * (n_catalogs - 1), min_catalogs
    )
    assert footprint == expected
