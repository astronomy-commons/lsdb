import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest

from lsdb.cutouts import (
    CutoutArray,
    CutoutRef,
    CutoutSeries,
    InMemoryImageStore,
    nestedframe_html,
    register_ipython_formatter,
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
    series = make_series(8, store)
    text = repr(series)
    assert "img1[0:3, 0:3]" in text
    assert "*" in text  # the star
    assert f"{8 - MAX_RENDERED} more cutouts <not rendered in preview>" in text
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
    series = make_series(8, store)
    html = series._repr_html_()
    assert html.count("<img") == MAX_RENDERED
    assert html.count("not rendered in preview") == 8 - MAX_RENDERED
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


def test_nestedframe_html(store):
    array = CutoutArray.from_arrays(
        ["img1"] * 8, x0=range(8), y0=range(8), width=[3] * 8, height=[3] * 8, store=store
    )
    frame = npd.NestedFrame({"a": range(8), "label": ["<b>bold</b>"] * 8, "cutouts": pd.Series(array)})
    html = nestedframe_html(frame)
    assert html.count("<img") == MAX_RENDERED
    assert html.count("not rendered in preview") == 8 - MAX_RENDERED
    # Other columns are escaped even though escape=False is used for the img tags
    assert "<b>bold</b>" not in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html


def test_nestedframe_html_truncates_rows(store):
    array = CutoutArray.from_arrays(
        ["img1"] * 15, x0=range(15), y0=range(15), width=[3] * 15, height=[3] * 15, store=store
    )
    frame = npd.NestedFrame({"a": range(15), "cutouts": pd.Series(array)})
    html = nestedframe_html(frame, max_rows=10)
    assert html.count("<img") == MAX_RENDERED
    assert "... 5 more rows" in html


def test_nestedframe_html_without_cutouts_is_default():
    frame = npd.NestedFrame({"a": [1, 2, 3]})
    assert nestedframe_html(frame) == frame._repr_html_()


def test_register_ipython_formatter_no_op_outside_ipython():
    # Running under pytest there is no IPython kernel; this must not raise.
    register_ipython_formatter()


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
