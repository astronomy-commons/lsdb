import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from nested_pandas import TensorArray, TensorDtype, TensorSeries

from lsdb.images import ImageDtype, ImageSeries


@pytest.fixture
def image_array():
    return TensorArray._from_sequence(
        [np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32)],
        dtype=ImageDtype("float32", shape=(2, 2)),
    )


def test_image_dtype_string_roundtrip():
    dtype = ImageDtype("float32", shape=(25, 25))
    assert dtype.name == "image[float, (25, 25)]"
    assert ImageDtype.construct_from_string(dtype.name) == dtype
    assert pd.api.types.pandas_dtype(dtype.name) == dtype
    with pytest.raises(TypeError):
        ImageDtype.construct_from_string("tensor[float, (25, 25)]")


def test_image_dtype_is_tensor_dtype():
    dtype = ImageDtype("float32", ndim=2)
    assert isinstance(dtype, TensorDtype)
    assert not dtype.is_fixed_shape


def test_image_column_returns_image_series(image_array):
    nf = npd.NestedFrame({"a": [1, 2]})
    nf["img"] = image_array
    column = nf["img"]
    assert isinstance(column, ImageSeries)
    assert column.tensor_shape == (2, 2)
    np.testing.assert_array_equal(column.to_image_stack()[0], np.ones((2, 2)))


def test_plain_tensor_column_stays_tensor_series():
    nf = npd.NestedFrame({"a": [1, 2]})
    nf["tensor"] = TensorArray.from_stack(np.zeros((2, 3, 3), np.float32))
    column = nf["tensor"]
    assert isinstance(column, TensorSeries)
    assert not isinstance(column, ImageSeries)


def test_image_dtype_survives_operations(image_array):
    taken = image_array.take([1, 0])
    assert isinstance(taken.dtype, ImageDtype)
    concatenated = TensorArray._concat_same_type([image_array, image_array])
    assert isinstance(concatenated.dtype, ImageDtype)
