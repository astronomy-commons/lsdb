"""Round-trip tests for nested-pandas tensor and image columns through HATS catalogs."""

import numpy as np
import numpy.testing as npt
import pytest
from nested_pandas import ImageDtype, ImageSeries, TensorArray, TensorDtype, TensorSeries

import lsdb


def _with_column(catalog, name, dtype):
    """Add a deterministic tensor/image column to every partition of a catalog."""

    def add_column(partition):
        rng = np.random.default_rng(len(partition))
        if dtype.is_fixed_shape:
            tensors = list(rng.normal(size=(len(partition), *dtype.shape)).astype(dtype.np_value_dtype))
        else:
            tensors = [
                rng.normal(size=(2 + row % 3, 4)).astype(dtype.np_value_dtype)
                for row in range(len(partition))
            ]
        partition[name] = TensorArray._from_sequence(tensors, dtype=dtype)
        return partition

    meta = catalog.meta.copy()
    meta[name] = TensorArray._from_sequence([], dtype=dtype)
    return catalog.map_partitions(add_column, meta=meta)


@pytest.mark.parametrize(
    "dtype",
    [
        TensorDtype("float32", shape=(4, 4)),
        TensorDtype("float64", ndim=2),
        ImageDtype("float32", shape=(4, 4)),
        ImageDtype("float32", ndim=2),
    ],
    ids=["tensor-fixed", "tensor-ragged", "image-fixed", "image-ragged"],
)
def test_catalog_roundtrip_preserves_tensor_columns(small_sky_order1_catalog, tmp_path, dtype):
    catalog = _with_column(small_sky_order1_catalog, "tensor", dtype)
    assert catalog.dtypes["tensor"] == dtype
    expected = catalog.compute()

    path = tmp_path / "tensor_catalog"
    catalog.write_catalog(path, catalog_name="tensor_catalog")
    reopened = lsdb.open_catalog(path)

    # The dask meta (schema-only, before any compute) keeps the dtype.
    assert reopened.dtypes["tensor"] == dtype

    result = reopened.compute()
    assert result["tensor"].dtype == dtype
    expected_class = ImageSeries if isinstance(dtype, ImageDtype) else TensorSeries
    assert isinstance(result["tensor"], expected_class)

    npt.assert_array_equal(result["tensor"].array.isna(), expected["tensor"].array.isna())
    for got, want in zip(result["tensor"].array, expected["tensor"].array, strict=True):
        npt.assert_array_equal(got, want)


def test_catalog_roundtrip_with_missing_rows(small_sky_order1_catalog, tmp_path):
    dtype = ImageDtype("float32", shape=(4, 4))

    def add_column(partition):
        tensors = [
            None if row % 3 == 0 else np.full((4, 4), row, dtype=np.float32)
            for row in range(len(partition))
        ]
        partition["img"] = TensorArray._from_sequence(tensors, dtype=dtype)
        return partition

    meta = small_sky_order1_catalog.meta.copy()
    meta["img"] = TensorArray._from_sequence([], dtype=dtype)
    catalog = small_sky_order1_catalog.map_partitions(add_column, meta=meta)
    expected = catalog.compute()
    assert expected["img"].array.isna().any()

    path = tmp_path / "img_catalog"
    catalog.write_catalog(path, catalog_name="img_catalog")
    result = lsdb.open_catalog(path).compute()

    assert result["img"].dtype == dtype
    npt.assert_array_equal(result["img"].array.isna(), expected["img"].array.isna())
    npt.assert_array_equal(
        result["img"].to_image_stack(), expected["img"].to_image_stack()
    )
