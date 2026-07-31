import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest

from lsdb.cutouts import (
    CutoutArray,
    CutoutRef,
    CutoutSeries,
    InMemoryImageStore,
)
from lsdb.cutouts.display import MAX_RENDERED, render_png_base64, series_html, series_repr


@pytest.fixture
def store():
    return InMemoryImageStore({"img1": np.arange(400.0).reshape(20, 20)})


def make_series(n, store, with_na=False):
    scalars = [CutoutRef("img1", i, i, 3, 3, store=store) for i in range(n)]
    if with_na:
        scalars[1] = None
    return CutoutSeries(CutoutArray._from_sequence(scalars).with_store(store))


def test_render_png_base64():
    png = render_png_base64(np.arange(9.0).reshape(3, 3))
    assert isinstance(png, str) and len(png) > 0
    # Constant and all-nan images do not crash
    assert render_png_base64(np.zeros((3, 3))) is not None
    assert render_png_base64(np.full((3, 3), np.nan)) is not None


def test_series_repr(store):
    n = MAX_RENDERED + 3
    series = make_series(n, store)
    text = repr(series)
    assert "img1[0:3, 0:3]" in text
    assert "*" in text  # the star
    assert "3 more cutouts <not rendered in preview>" in text
    assert "dtype: cutout" in text
    assert "InMemoryImageStore" in text


def test_series_repr_short_and_na(store):
    series = make_series(3, store, with_na=True)
    text = repr(series)
    assert "<NA>" in text
    assert "more cutouts" not in text


def test_series_repr_does_not_touch_pixels(store):
    class ExplodingStore(InMemoryImageStore):
        def get_image(self, image_id):
            raise AssertionError("text repr must not fetch pixels")

    exploding = ExplodingStore({"img1": np.zeros((1, 1))})
    series = make_series(3, store).with_store(exploding)
    assert "img1[0:3, 0:3]" in repr(series)


def test_series_repr_non_cutout_falls_back():
    series = CutoutSeries([1, 2, 3])
    assert "int64" in repr(series)
    assert series._repr_html_() is None


def test_series_html(store):
    n = MAX_RENDERED + 3
    series = make_series(n, store)
    html = series._repr_html_()
    assert html.count("<img") == MAX_RENDERED
    assert html.count("not rendered in preview") == 3
    assert "data:image/png;base64," in html


def test_series_html_no_store(store):
    series = make_series(2, store).with_store(None)
    html = series._repr_html_()
    assert "<img" not in html
    assert "no image store attached" in html


def test_series_html_na(store):
    series = make_series(3, store, with_na=True)
    html = series._repr_html_()
    assert "&lt;NA&gt;" in html
    assert html.count("<img") == 2


def test_nestedframe_repr_renders_thumbnails(store):
    array = CutoutArray.from_arrays(
        ["img1"] * 4, x0=range(4), y0=range(4), width=[3] * 4, height=[3] * 4, store=store
    )
    frame = npd.NestedFrame({"a": range(4), "cutouts": pd.Series(array)})
    html = frame._repr_html_()
    # Rendered natively by NestedFrame via the registered cell formatter
    assert html.count("<img") == 4
    assert "data:image/png;base64," in html


def test_nestedframe_repr_truncation_bounds_thumbnails(store):
    n = 40
    array = CutoutArray.from_arrays(
        ["img1"] * n,
        x0=[i % 15 for i in range(n)],
        y0=[i % 15 for i in range(n)],
        width=[3] * n,
        height=[3] * n,
        store=store,
    )
    frame = npd.NestedFrame({"a": range(n), "cutouts": pd.Series(array)})
    with pd.option_context("display.max_rows", 10, "display.min_rows", 5):
        html = frame._repr_html_()
    # pandas row truncation bounds the number of thumbnails rendered
    assert 0 < html.count("<img") <= 6


def test_nestedframe_repr_without_cutouts_unchanged():
    frame = npd.NestedFrame({"a": [1, 2, 3]})
    html = frame._repr_html_()
    assert "<img" not in html
    assert "<table" in html


def test_cutout_cell_html_handles_na():
    from lsdb.cutouts import cutout_cell_html

    assert cutout_cell_html(pd.NA) == "&lt;NA&gt;"
    assert cutout_cell_html(None) == "&lt;NA&gt;"


def test_ref_repr_html(store):
    ref = CutoutRef("img1", 2, 3, 5, 4, store=store)
    html = ref._repr_html_()
    assert "<img" in html
    assert "img1[3:7, 2:7]" in html
    assert "(4 x 5)" in html


def test_ref_repr_html_no_store_falls_back():
    ref = CutoutRef("img1", 2, 3, 5, 4)
    assert ref._repr_html_() is None
    assert "CutoutRef" in repr(ref)


def test_nestedframe_repr_nested_and_cutout_columns(store):
    array = CutoutArray.from_arrays(
        ["img1"] * 3, x0=[0, 1, 2], y0=[0, 1, 2], width=[3] * 3, height=[3] * 3, store=store
    )
    frame = npd.NestedFrame(
        {"a": [1, 2, 3], "flux": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], "cutouts": pd.Series(array)}
    ).nest_lists(columns=["flux"], name="lightcurve")
    html = frame._repr_html_()
    # Nested sub-tables and cutout thumbnails render side by side
    assert "_DataFrameWrapperForRepresentation" not in html
    assert "+2 rows" in html  # nested-pandas sub-table footer for the 3-row cell
    assert html.count("<img") == 3
