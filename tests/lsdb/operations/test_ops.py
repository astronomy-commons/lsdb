import nested_pandas as npd
import pandas as pd
import pytest
from dask.utils import funcname
from hats.pixel_math import HealpixPixel

from lsdb.operations.lsdb_ops import (
    AlignAndApply,
    EmptyOperation,
    FromHealpixMap,
    MapPartitions,
    SelectColumns,
    SelectPixels,
    _coerce_to_frame,
    _coerce_to_meta,
    _normalize_meta,
    map_parts_meta,
)


def _make_frame(pixel):
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
    def bad_func(pixel):  # pylint: disable=unused-argument
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
    meta, is_df_type = _coerce_to_meta(df)

    assert isinstance(meta, npd.NestedFrame)
    assert len(meta) == 0
    assert list(meta.columns) == ["a", "b"]
    assert is_df_type


def test_coerce_to_meta_series_keeps_name_and_marks_non_df():
    series = pd.Series([1.0, 2.0], name="flux")
    meta, is_df_type = _coerce_to_meta(series)

    assert isinstance(meta, npd.NestedFrame)
    assert len(meta) == 0
    assert list(meta.columns) == ["flux"]
    assert not is_df_type


@pytest.mark.parametrize(
    "values, expected_dtype_kind",
    [
        ([1, 2, 3], "i"),
        ([1.0, 2.0], "f"),
        (["x", "y"], "O"),
    ],
)
def test_coerce_to_meta_dict_of_lists_infers_dtype_from_first_element(values, expected_dtype_kind):
    meta, is_df_type = _coerce_to_meta({"col": values})

    assert len(meta) == 0
    assert meta["col"].dtype.kind == expected_dtype_kind
    assert is_df_type


@pytest.mark.parametrize("empty_collection", [[], ()])
def test_coerce_to_meta_empty_list_or_tuple_uses_object_dtype(empty_collection):
    meta, is_df_type = _coerce_to_meta(empty_collection)

    assert len(meta) == 0
    assert meta["result"].dtype == object
    assert not is_df_type


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
    meta, is_df_type = _coerce_to_meta(scalar)

    assert len(meta) == 0
    assert list(meta.columns) == ["result"]
    assert meta["result"].dtype == expected_dtype
    assert not is_df_type


def test_coerce_to_meta_unsafe_pandas_type():

    class Custom:  # pylint: disable=too-few-public-methods
        pass

    # result should be a pd.series with dtype=object
    meta, is_df_type = _coerce_to_meta(Custom())
    assert len(meta) == 0
    assert list(meta.columns) == ["result"]
    assert meta["result"].dtype == object
    assert not is_df_type


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
    result, is_df_type = _normalize_meta(nf)

    assert result is nf
    assert is_df_type


@pytest.mark.parametrize(
    "meta",
    [
        pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")}),
        {"a": "int64", "b": "float64"},
        [("a", "int64"), ("b", "float64")],
    ],
)
def test_normalize_meta_accepted_formats_produce_equivalent_nestedframe(meta):
    result, is_df_type = _normalize_meta(meta)

    assert isinstance(result, npd.NestedFrame)
    assert list(result.columns) == ["a", "b"]
    assert dict(result.dtypes) == {"a": pd.Series(dtype="int64").dtype, "b": pd.Series(dtype="float64").dtype}
    assert is_df_type


def test_normalize_meta_series_marks_non_df_output():
    series_meta = pd.Series(dtype="float64", name="flux")
    result, is_df_type = _normalize_meta(series_meta)

    assert isinstance(result, npd.NestedFrame)
    assert len(result) == 0
    assert list(result.columns) == ["flux"]
    assert result["flux"].dtype == series_meta.dtype
    assert not is_df_type


@pytest.mark.parametrize("bad_meta", ["not valid", 5, [1, 2, 3]])
def test_normalize_meta_invalid_input_raises_value_error(bad_meta):
    with pytest.raises(ValueError, match="meta must be a DataFrame"):
        _normalize_meta(bad_meta)


def _base_mp():
    meta = npd.NestedFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})
    return EmptyOperation(meta=meta)


def _func(df):
    return df


def test_map_partitions_class_func_conflict():
    # SelectColumns sets class_func, so passing a non-None func conflicts.
    with pytest.raises(ValueError, match="Cannot specify func for MapPartitions when class_func is set"):
        SelectColumns(_base_mp(), _func, ["a"])

    # Correct usage: func=None lets class_func take over.
    op = SelectColumns(_base_mp(), None, ["a"])
    assert op.func is SelectColumns.class_func


class _IncludePixelOp(MapPartitions):
    class_include_pixels = True


@pytest.mark.parametrize("include_pixel", [False, None])
def test_map_partitions_class_include_pixels_conflict(include_pixel):
    kwargs = {} if include_pixel is None else {"include_pixel": include_pixel}

    with pytest.raises(
        ValueError, match="Cannot specify include_pixel for MapPartitions when class_include_pixels is set"
    ):
        _IncludePixelOp(_base_mp(), _func, **kwargs)

    # Correct usage: include_pixel matches class_include_pixels.
    op = _IncludePixelOp(_base_mp(), _func, include_pixel=True)
    assert op.include_pixel is True


def test_map_partitions_name_and_dependencies():
    base = _base_mp()
    op = MapPartitions(base, _func)

    assert op.name == f"MapPartitions({funcname(_func)}, {base.name})"
    assert op.dependencies == [base]


def _base_select():

    def func(pixel):  # pylint: disable=unused-argument
        return pd.DataFrame({"a": [1], "b": [1.0]})

    pixels = [HealpixPixel(0, 0), HealpixPixel(0, 1)]
    return FromHealpixMap(func, pixels)


def test_select_columns_column_selector_returns_first_arg():
    op = SelectColumns(_base_select(), None, ["a", "b"])

    assert op.column_selector == ["a", "b"]


def test_select_pixels_name_and_dependencies():
    base = _base_select()
    op = SelectPixels(base, base.healpix_pixels[:1])

    assert op.name == f"SelectPixels({base.name})"
    assert op.dependencies == [base]


def test_select_pixels_build_raises_for_pixel_not_in_base():
    base = _base_select()
    op = SelectPixels(base, [HealpixPixel(5, 5)])

    with pytest.raises(ValueError, match="not found in operation"):
        op.build()


def _meta_aa():
    return npd.NestedFrame({"a": pd.Series(dtype="int64")})


def _func_aa():
    return pd.DataFrame({"a": [1]})


class _MockCatalog:  # pylint: disable=too-few-public-methods
    """Stand-in for a HealpixDataset, exposing only what AlignAndApply needs."""

    def __init__(self, operation):
        self._operation = operation


def test_align_and_apply_mismatched_input_lengths_raises_value_error():
    with pytest.raises(ValueError, match="Incorrect Align and Apply Setup"):
        AlignAndApply(input_cats=[], pixel_lists=[[]], func=_func_aa, meta=_meta_aa(), output_pixels=[])


def test_align_and_apply_dependencies_filters_out_none_inputs():
    op1 = EmptyOperation(meta=_meta_aa())
    op2 = EmptyOperation(meta=_meta_aa())
    input_cats = [_MockCatalog(op1), None, _MockCatalog(op2)]

    aa = AlignAndApply(
        input_cats=input_cats, pixel_lists=[[], [], []], func=_func_aa, meta=_meta_aa(), output_pixels=[]
    )

    assert aa.dependencies == [op1, op2]


def test_align_and_apply_name_includes_func_and_input_op_names_or_none():
    op1 = EmptyOperation(meta=_meta_aa())
    op2 = EmptyOperation(meta=_meta_aa())
    input_cats = [_MockCatalog(op1), None, _MockCatalog(op2)]

    aa = AlignAndApply(
        input_cats=input_cats, pixel_lists=[[], [], []], func=_func_aa, meta=_meta_aa(), output_pixels=[]
    )

    assert aa.name == f"AlignAndApply({funcname(_func_aa)}, {op1.name}, None, {op2.name})"
