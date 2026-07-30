import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from lsdb.cutouts import (
    CUTOUT_ARROW_TYPE,
    ChainImageStore,
    CutoutArray,
    CutoutDtype,
    CutoutRef,
    CutoutSeries,
    InMemoryImageStore,
    merge_stores,
)


@pytest.fixture
def image():
    return np.arange(400.0).reshape(20, 20)


@pytest.fixture
def wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10, 10]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [180.0, -30.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def store(image, wcs):
    return InMemoryImageStore({"img1": image}, wcs={"img1": wcs})


@pytest.fixture
def array(store):
    return CutoutArray.from_arrays(
        image_id=["img1", "img1", "img1"],
        x0=[0, 2, 5],
        y0=[0, 2, 4],
        width=[3, 4, 3],
        height=[3, 4, 5],
        store=store,
    )


def test_construction_and_dtype(array):
    assert len(array) == 3
    assert isinstance(array.dtype, CutoutDtype)
    assert array.dtype.name == "cutout"
    assert array.nbytes > 0


def test_series_construction(array):
    series = pd.Series(array)
    assert series.dtype == "cutout"
    assert len(series) == 3


def test_from_sequence_of_refs(array):
    refs = list(array)
    rebuilt = CutoutArray._from_sequence(refs)
    assert len(rebuilt) == 3
    # Store is recovered from the refs themselves
    assert rebuilt.store is array.store
    assert rebuilt[1] == array[1]


def test_from_sequence_with_missing(store):
    array = CutoutArray._from_sequence([CutoutRef("img1", 0, 0, 2, 2, store=store), None, pd.NA])
    assert len(array) == 3
    np.testing.assert_array_equal(array.isna(), [False, True, True])
    assert array[1] is pd.NA


def test_pd_array_constructor(store):
    values = pd.array([CutoutRef("img1", 0, 0, 2, 2, store=store)], dtype="cutout")
    assert isinstance(values, CutoutArray)


def test_getitem_scalar(array, image):
    ref = array[0]
    assert isinstance(ref, CutoutRef)
    assert ref.shape == (3, 3)
    np.testing.assert_array_equal(ref.data, image[0:3, 0:3])
    # Negative indexing
    assert array[-1] == array[2]


def test_getitem_slice_mask_fancy(array):
    assert len(array[1:]) == 2
    assert array[1:][0] == array[1]
    masked = array[np.array([True, False, True])]
    assert len(masked) == 2
    assert masked[1] == array[2]
    taken = array[np.array([2, 0])]
    assert taken[0] == array[2]
    # Store propagates through all paths
    assert masked.store is array.store
    assert taken.store is array.store
    assert array[1:].store is array.store


def test_take_allow_fill(array):
    taken = array.take(np.array([0, -1, 2]), allow_fill=True)
    assert len(taken) == 3
    assert taken[1] is pd.NA
    assert taken[2] == array[2]
    with pytest.raises(ValueError):
        array.take(np.array([0, -2]), allow_fill=True)


def test_take_out_of_bounds(array):
    with pytest.raises(IndexError):
        array.take(np.array([5]))


def test_zero_copy_views(array, image):
    # Rendered cutouts are views into the stored image
    assert np.shares_memory(array[0].data, image)
    # Overlapping cutouts share pixels with each other
    assert np.shares_memory(array[0].data, array[1].data)


def test_no_store_raises(array):
    detached = array.with_store(None)
    assert detached.store is None
    with pytest.raises(ValueError, match="no image store"):
        _ = detached[0].data
    reattached = detached.with_store(array.store)
    np.testing.assert_array_equal(reattached[0].data, array[0].data)


def test_concat_merges_stores(image):
    image2 = np.ones((10, 10))
    store_a = InMemoryImageStore({"img1": image})
    store_b = InMemoryImageStore({"img2": image2})
    array_a = CutoutArray.from_arrays(["img1"], [0], [0], [2], [2], store=store_a)
    array_b = CutoutArray.from_arrays(["img2"], [3], [3], [2], [2], store=store_b)
    series = pd.concat([pd.Series(array_a), pd.Series(array_b)], ignore_index=True)
    combined = series.array
    assert isinstance(combined, CutoutArray)
    assert isinstance(combined.store, ChainImageStore)
    np.testing.assert_array_equal(series.iloc[0].data, image[0:2, 0:2])
    np.testing.assert_array_equal(series.iloc[1].data, image2[3:5, 3:5])


def test_concat_same_store_not_chained(array):
    series = pd.concat([pd.Series(array), pd.Series(array)], ignore_index=True)
    assert series.array.store is array.store


def test_merge_stores_flattens_chains(store):
    other = InMemoryImageStore({"img2": np.zeros((5, 5))})
    chain = merge_stores([store, other])
    rechained = merge_stores([chain, store, None])
    assert isinstance(rechained, ChainImageStore)
    assert len(rechained.stores) == 2
    assert merge_stores([None, None]) is None
    assert merge_stores([store, store]) is store


def test_arrow_round_trip(array):
    arrow_array = pa.array(array)
    assert arrow_array.type == CUTOUT_ARROW_TYPE
    rebuilt = CutoutDtype().__from_arrow__(arrow_array)
    assert isinstance(rebuilt, CutoutArray)
    assert rebuilt.store is None
    assert rebuilt.with_store(array.store)[0] == array[0]


def test_parquet_round_trip(tmp_path, array):
    frame = pd.DataFrame({"id": [1, 2, 3], "cutouts": pd.Series(array)})
    path = tmp_path / "cutouts.parquet"
    frame.to_parquet(path)
    read = pd.read_parquet(path)
    assert read["cutouts"].dtype == "cutout"
    restored = read["cutouts"].cutout.with_store(array.store)
    np.testing.assert_array_equal(restored.iloc[0].data, array[0].data)


def test_descriptor_frame(array):
    frame = array.to_descriptor_frame()
    assert list(frame.columns) == ["image_id", "x0", "y0", "width", "height"]
    assert frame["x0"].tolist() == [0, 2, 5]
    assert frame["height"].tolist() == [3, 4, 5]


def test_accessor_on_plain_series(array, image):
    series = pd.Series(array)
    assert type(series) is pd.Series
    images = series.cutout.to_images()
    np.testing.assert_array_equal(images[0], image[0:3, 0:3])
    with pytest.raises(AttributeError):
        pd.Series([1, 2, 3]).cutout  # pylint: disable=expression-not-assigned


def test_to_image_stack(store, image):
    array = CutoutArray.from_arrays(
        ["img1", "img1"], x0=[0, 4], y0=[0, 4], width=[3, 3], height=[3, 3], store=store
    )
    stack = pd.Series(array).cutout.to_image_stack()
    assert stack.shape == (2, 3, 3)
    np.testing.assert_array_equal(stack[1], image[4:7, 4:7])


def test_to_image_stack_errors(array, store):
    with pytest.raises(ValueError, match="differing shapes"):
        pd.Series(array).cutout.to_image_stack()
    with_na = CutoutArray._from_sequence([CutoutRef("img1", 0, 0, 2, 2, store=store), None])
    with pytest.raises(ValueError, match="missing"):
        pd.Series(with_na).cutout.to_image_stack()


@pytest.mark.parametrize("x0,y0,width,height", [(2, 3, 5, 7), (2, 3, 4, 6), (0, 0, 3, 3)])
def test_to_cutout2d_matches_slices(store, image, wcs, x0, y0, width, height):
    array = CutoutArray.from_arrays(["img1"], [x0], [y0], [width], [height], store=store)
    cutout = pd.Series(array).cutout.to_cutout2d()[0]
    assert isinstance(cutout, Cutout2D)
    np.testing.assert_array_equal(cutout.data, image[y0 : y0 + height, x0 : x0 + width])
    # WCS is adjusted to the cutout frame: pixel (0, 0) of the cutout
    # is pixel (x0, y0) of the parent image
    parent_coord = wcs.pixel_to_world(x0, y0)
    cutout_coord = cutout.wcs.pixel_to_world(0, 0)
    assert parent_coord.separation(cutout_coord).arcsec < 1e-8


def test_nestedframe_returns_cutout_series(array):
    ndf = npd.NestedFrame({"a": [1, 2, 3], "cutouts": pd.Series(array)})
    series = ndf["cutouts"]
    assert type(series) is CutoutSeries
    cutouts = series.to_cutout2d()
    assert len(cutouts) == 3
    # Non-cutout columns are unaffected
    assert type(ndf["a"]) is pd.Series


def test_cutout_series_methods_gated():
    series = CutoutSeries([1, 2, 3])
    with pytest.raises(TypeError, match="cutout"):
        series.to_images()


def test_row_filtering_keeps_rendering(array):
    ndf = npd.NestedFrame({"a": [1, 2, 3], "cutouts": pd.Series(array)})
    filtered = ndf.query("a > 1")
    assert len(filtered) == 2
    series = filtered["cutouts"]
    assert type(series) is CutoutSeries
    assert series.to_images()[0].shape == (4, 4)


def test_to_cutout2d_copy_parameter(store, image):
    ref = CutoutRef("img1", 2, 3, 5, 4, store=store)
    view_cutout = ref.to_cutout2d()
    assert np.shares_memory(view_cutout.data, image)
    detached = ref.to_cutout2d(copy=True)
    assert not np.shares_memory(detached.data, image)
    np.testing.assert_array_equal(detached.data, view_cutout.data)
    # Series-level passthrough
    series = pd.Series(CutoutArray._from_sequence([ref]))
    assert not np.shares_memory(series.cutout.to_cutout2d(copy=True)[0].data, image)
