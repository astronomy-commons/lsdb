import nested_pandas as npd
import pandas as pd
import pytest
from hats.pixel_math import HealpixPixel

from lsdb.operations.lsdb_ops import (
    EmptyOperation,
    FromHealpixMap,
    _coerce_to_frame,
    _coerce_to_meta,
    _normalize_meta,
    map_parts_meta,
)


def _make_frame(pixel, *args, **kwargs):
    """A simple FromHealpixMap function that returns a populated DataFrame."""
    return pd.DataFrame({"a": [pixel.pixel, pixel.pixel + 1], "b": [float(pixel.order), float(pixel.order)]})


def _base_meta():
    return npd.NestedFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})


def test_meta_returns_provided_meta_directly():
    provided_meta = npd.NestedFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})
    op = FromHealpixMap(_make_frame, [HealpixPixel(0, 0)], meta=provided_meta)
    assert op.meta is provided_meta


def test_meta_inferred_with_map_kwargs_raises_type_error():
    def func(pixel, scale):
        return pd.DataFrame({"value": [pixel.pixel * scale]})

    pixels = [HealpixPixel(0, 0), HealpixPixel(0, 1)]
    op = FromHealpixMap(func, pixels, map_kwargs={"scale": [1, 2]})

    # map_kwargs are per-pixel and only applied in `build`, not when
    # inferring meta, so the first pixel's func call is missing `scale`.
    with pytest.raises(TypeError):
        _ = op.meta


@pytest.mark.parametrize(
    "bad_result",
    [
        pd.Series([1, 2, 3]),
        {"a": [1, 2, 3]},
        [1, 2, 3],
        "not a dataframe",
        42,
        None,
    ],
)
def test_meta_raises_value_error_for_non_dataframe_result(bad_result):
    def bad_func(pixel, *args, **kwargs):
        return bad_result

    op = FromHealpixMap(bad_func, [HealpixPixel(0, 0)])

    with pytest.raises(ValueError, match="FromMap function must return a pandas DataFrame"):
        _ = op.meta


def test_map_parts_meta_wraps_error_from_function_assuming_nonempty_data():
    def func(df):
        return df.iloc[0]

    with pytest.raises(ValueError, match="Cannot infer meta for MapPartitions") as exc_info:
        map_parts_meta(func, _base_meta())

    assert isinstance(exc_info.value.__cause__, IndexError)


def test_coerce_to_meta_plain_dataframe_is_converted_and_emptied():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    meta = _coerce_to_meta(df)

    assert isinstance(meta, npd.NestedFrame)
    assert len(meta) == 0
    assert list(meta.columns) == ["a", "b"]


@pytest.mark.parametrize(
    "values, expected_dtype_kind",
    [
        ([1, 2, 3], "i"),
        ([1.0, 2.0], "f"),
        (["x", "y"], "O"),
    ],
)
def test_coerce_to_meta_dict_of_lists_infers_dtype_from_first_element(values, expected_dtype_kind):
    meta = _coerce_to_meta({"col": values})

    assert len(meta) == 0
    assert meta["col"].dtype.kind == expected_dtype_kind


@pytest.mark.parametrize("empty_collection", [[], ()])
def test_coerce_to_meta_empty_list_or_tuple_uses_object_dtype(empty_collection):
    meta = _coerce_to_meta(empty_collection)

    assert len(meta) == 0
    assert meta["result"].dtype == object


@pytest.mark.parametrize(
    "scalar, expected_dtype",
    [
        (5, "int64"),
        (5.0, "float64"),
        (True, "bool"),
        ("hello", "object"),
    ],
)
def test_coerce_to_meta_scalar_infers_dtype_from_type(scalar, expected_dtype):
    meta = _coerce_to_meta(scalar)

    assert len(meta) == 0
    assert list(meta.columns) == ["result"]
    assert meta["result"].dtype == expected_dtype


def test_coerce_to_meta_unsupported_scalar_type_raises_type_error():
    # KNOWN ISSUE: an object of a type pandas doesn't recognize as a dtype
    # (e.g. a custom class instance, or a set) falls through to the scalar
    # branch and raises an unwrapped TypeError from `_coerce_to_meta`. This
    # is *not* caught by `map_parts_meta`'s try/except, since the call to
    # `_coerce_to_meta` happens after that block, so it surfaces directly.
    class Custom:
        pass

    with pytest.raises(TypeError, match="not understood"):
        _coerce_to_meta(Custom())


@pytest.mark.parametrize("values", [[1, 2, 3], (1, 2, 3)])
def test_coerce_to_frame_list_or_tuple_creates_one_row_per_element(values):
    frame = _coerce_to_frame(values)

    assert isinstance(frame, npd.NestedFrame)
    assert list(frame.columns) == ["result"]
    assert frame["result"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("empty_collection", [[], ()])
def test_coerce_to_frame_empty_list_or_tuple_produces_empty_frame(empty_collection):
    frame = _coerce_to_frame(empty_collection)

    assert list(frame.columns) == ["result"]
    assert len(frame) == 0


@pytest.mark.parametrize("scalar", [5, "hello", None])
def test_coerce_to_frame_scalar_creates_single_row(scalar):
    frame = _coerce_to_frame(scalar)

    assert list(frame.columns) == ["result"]
    assert len(frame) == 1
    assert frame["result"].iloc[0] == scalar or frame["result"].iloc[0] is scalar


def test_normalize_meta_nestedframe_passthrough():
    nf = npd.NestedFrame({"a": pd.Series(dtype="int64")})

    assert _normalize_meta(nf) is nf


@pytest.mark.parametrize(
    "meta",
    [
        pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")}),
        {"a": "int64", "b": "float64"},
        [("a", "int64"), ("b", "float64")],
    ],
)
def test_normalize_meta_accepted_formats_produce_equivalent_nestedframe(meta):
    result = _normalize_meta(meta)

    assert isinstance(result, npd.NestedFrame)
    assert list(result.columns) == ["a", "b"]
    assert dict(result.dtypes) == {"a": pd.Series(dtype="int64").dtype, "b": pd.Series(dtype="float64").dtype}


@pytest.mark.parametrize("bad_meta", ["not valid", 5, [1, 2, 3]])
def test_normalize_meta_invalid_input_raises_value_error(bad_meta):
    with pytest.raises(ValueError, match="meta must be a DataFrame"):
        _normalize_meta(bad_meta)
