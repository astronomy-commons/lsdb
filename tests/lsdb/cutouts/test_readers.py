import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from lsdb.catalog.image_catalog import wcs_to_header_string
from lsdb.cutouts import (
    CatalogImageStore,
    CutoutArray,
    FitsImageReader,
    ImageReader,
    ZarrImageReader,
    get_image_reader,
    register_image_reader,
    resolve_image_format,
)


def make_wcs(ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [25, 25]
    wcs.wcs.cdelt = [-0.005, 0.005]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def fits_image(tmp_path):
    data = np.arange(2500.0).reshape(50, 50)
    path = tmp_path / "v1.fits"
    fits.PrimaryHDU(data=data).writeto(path)
    return str(path), data


@pytest.fixture
def image_rows(fits_image):
    path, _ = fits_image
    return pd.DataFrame(
        {
            "image_id": ["v1"],
            "path": [path],
            "width": [50],
            "height": [50],
            "wcs": [wcs_to_header_string(make_wcs(180.0, -30.0))],
        }
    )


def test_fits_reader(fits_image):
    path, data = fits_image
    read = FitsImageReader().read_image(path)
    np.testing.assert_array_equal(read, data)


def test_fits_reader_finds_image_hdu(tmp_path):
    data = np.ones((10, 10))
    path = tmp_path / "multi.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data)]).writeto(path)
    np.testing.assert_array_equal(FitsImageReader().read_image(str(path)), data)
    with pytest.raises(ValueError, match="No 2D image HDU"):
        empty = tmp_path / "empty.fits"
        fits.HDUList([fits.PrimaryHDU()]).writeto(empty)
        FitsImageReader().read_image(str(empty))


def test_zarr_reader(tmp_path):
    zarr = pytest.importorskip("zarr", reason="zarr not installed")
    data = np.arange(100.0).reshape(10, 10)
    path = str(tmp_path / "image.zarr")
    array = zarr.open(path, mode="w", shape=data.shape, dtype=data.dtype)
    array[:] = data
    np.testing.assert_array_equal(ZarrImageReader().read_image(path), data)


def test_zarr_reader_error_without_zarr(tmp_path):
    try:
        import zarr  # noqa: F401

        pytest.skip("zarr installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="requires the 'zarr' package"):
        ZarrImageReader().read_image(str(tmp_path / "image.zarr"))


def test_reader_registry():
    assert isinstance(get_image_reader("fits"), FitsImageReader)
    with pytest.raises(ValueError, match="No image reader registered"):
        get_image_reader("hdf5")

    class DummyReader(ImageReader):
        def read_image(self, path):
            return np.zeros((2, 2))

    register_image_reader("dummy", DummyReader())
    try:
        assert isinstance(get_image_reader("dummy"), DummyReader)
    finally:
        from lsdb.cutouts.readers import _READERS

        _READERS.pop("dummy")


def test_resolve_image_format():
    assert resolve_image_format("zarr", "fits", "x.fits") == "zarr"  # row wins
    assert resolve_image_format(None, "fits", "x.zarr") == "fits"  # catalog wins
    assert resolve_image_format(None, None, "s3://bucket/x.fits") == "fits"
    assert resolve_image_format(None, None, "x.fz") == "fits"
    assert resolve_image_format(None, None, "store.zarr") == "zarr"
    with pytest.raises(ValueError, match="Cannot determine image format"):
        resolve_image_format(None, None, "mystery.dat")


def test_catalog_image_store(image_rows, fits_image):
    _, data = fits_image
    store = CatalogImageStore(image_rows)
    assert "v1" in store
    assert "v2" not in store
    np.testing.assert_array_equal(store.get_image("v1"), data)
    # Cached: same object on second access
    assert store.get_image("v1") is store.get_image("v1")
    wcs = store.get_wcs("v1")
    assert wcs.has_celestial
    assert store.get_wcs("v1") is wcs
    with pytest.raises(KeyError, match="not found"):
        store.get_image("v2")


def test_catalog_image_store_deduplicates(image_rows):
    duplicated = pd.concat([image_rows, image_rows], ignore_index=True)
    store = CatalogImageStore(duplicated)
    assert store.get_image("v1").shape == (50, 50)


def test_catalog_image_store_format_column(image_rows):
    rows = image_rows.assign(format="fits", path=image_rows["path"].str.replace(".fits", ".weird"))
    # Row-level format wins over extension sniffing; file missing -> reader error, not format error
    store = CatalogImageStore(rows)
    with pytest.raises(FileNotFoundError):
        store.get_image("v1")


def test_cutout_column_renders_from_fits(image_rows, fits_image):
    _, data = fits_image
    store = CatalogImageStore(image_rows)
    array = CutoutArray.from_descriptors(["v1", "v1"], x0=[0, 10], y0=[0, 10], width=[5, 5], height=[5, 5])
    series = pd.Series(array.with_store(store))
    np.testing.assert_array_equal(series.iloc[1], data[10:15, 10:15])


def test_fits_reader_file_uri(fits_image):
    path, data = fits_image
    np.testing.assert_array_equal(FitsImageReader().read_image(f"file://{path}"), data)


@pytest.fixture
def compressed_fits_image(tmp_path):
    data = np.arange(10000.0, dtype=np.float32).reshape(100, 100)
    path = tmp_path / "compressed.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=data, name="IMAGE")]).writeto(path)
    return str(path), data


def test_fits_read_region_plain(fits_image):
    path, data = fits_image
    np.testing.assert_array_equal(FitsImageReader().read_region(path, 10, 20, 5, 15), data[10:20, 5:15])


def test_fits_read_region_compressed(compressed_fits_image):
    path, data = compressed_fits_image
    region = FitsImageReader().read_region(path, 30, 55, 40, 90)
    np.testing.assert_array_equal(region, data[30:55, 40:90])


def test_fits_read_region_clamps(compressed_fits_image):
    path, data = compressed_fits_image
    # Overhanging boxes clamp to the image; fully-outside boxes come back empty
    np.testing.assert_array_equal(FitsImageReader().read_region(path, -10, 5, 95, 200), data[0:5, 95:100])
    assert FitsImageReader().read_region(path, 200, 300, 0, 10).size == 0


def test_zarr_read_region(tmp_path):
    zarr = pytest.importorskip("zarr", reason="zarr not installed")
    data = np.arange(10000.0).reshape(100, 100)
    path = str(tmp_path / "image.zarr")
    array = zarr.open(path, mode="w", shape=data.shape, dtype=data.dtype, chunks=(10, 10))
    array[:] = data
    np.testing.assert_array_equal(ZarrImageReader().read_region(path, 15, 25, 35, 45), data[15:25, 35:45])


class CountingReader(ImageReader):
    """Test double that counts full vs region reads."""

    def __init__(self, data):
        self.data = data
        self.full_reads = 0
        self.region_reads = 0

    def read_image(self, path):
        self.full_reads += 1
        return self.data

    def read_region(self, path, y0, y1, x0, x1):
        self.region_reads += 1
        return self.data[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]


@pytest.fixture
def counting_store(image_rows):
    def make(read_mode="auto", full_read_threshold=3):
        reader = CountingReader(np.arange(2500.0).reshape(50, 50))
        register_image_reader("counting", reader)
        rows = image_rows.assign(format="counting")
        store = CatalogImageStore(rows, read_mode=read_mode, full_read_threshold=full_read_threshold)
        return store, reader

    yield make
    from lsdb.cutouts.readers import _READERS

    _READERS.pop("counting", None)


def test_store_region_mode_never_reads_full(counting_store):
    store, reader = counting_store(read_mode="region")
    for position in range(10):
        store.get_region("v1", position, position + 5, 0, 5)
    assert reader.full_reads == 0
    assert reader.region_reads == 10
    # Identical region requests are cached: one read for two requests
    store.get_region("v1", 40, 45, 40, 45)
    store.get_region("v1", 40, 45, 40, 45)
    assert reader.region_reads == 11


def test_store_full_mode_reads_once(counting_store):
    store, reader = counting_store(read_mode="full")
    for position in range(10):
        store.get_region("v1", position, position + 5, 0, 5)
    assert reader.full_reads == 1
    assert reader.region_reads == 0


def test_store_auto_mode_switches_to_full(counting_store):
    store, reader = counting_store(read_mode="auto", full_read_threshold=3)
    for position in range(10):
        store.get_region("v1", position, position + 5, 0, 5)
    # 3 region reads, then one full read serves everything after
    assert reader.region_reads == 3
    assert reader.full_reads == 1
    # Once full is cached, regions are views into it
    region = store.get_region("v1", 0, 5, 0, 5)
    assert np.shares_memory(region, store.get_image("v1"))


def test_store_read_mode_validation(image_rows):
    with pytest.raises(ValueError, match="read_mode"):
        CatalogImageStore(image_rows, read_mode="sometimes")


def test_cutout_data_uses_region_reads(image_rows, counting_store):
    store, reader = counting_store(read_mode="region")
    array = CutoutArray.from_descriptors(["v1"], x0=[10], y0=[20], width=[5], height=[7], store=store)
    data = array[0]
    assert data.shape == (7, 5)
    assert reader.region_reads == 1
    assert reader.full_reads == 0


def test_to_images_plans_reads_upfront(image_rows, counting_store):
    # 10 cutouts from one image, threshold 3: planning does ONE full read, zero region reads
    store, reader = counting_store(read_mode="auto", full_read_threshold=3)
    array = CutoutArray.from_descriptors(
        ["v1"] * 10, x0=list(range(10)), y0=list(range(10)), width=[5] * 10, height=[5] * 10, store=store
    )
    images = array.to_stack()
    assert len(images) == 10
    assert reader.full_reads == 1
    assert reader.region_reads == 0


def test_to_images_small_batch_stays_regional(image_rows, counting_store):
    # 2 cutouts, threshold 3: planning leaves them as cheap region reads
    store, reader = counting_store(read_mode="auto", full_read_threshold=3)
    array = CutoutArray.from_descriptors(
        ["v1", "v1"], x0=[0, 10], y0=[0, 10], width=[5, 5], height=[5, 5], store=store
    )
    array.to_stack()
    assert reader.full_reads == 0
    assert reader.region_reads == 2


def test_plan_reads_respects_region_mode(counting_store):
    store, reader = counting_store(read_mode="region")
    store.plan_reads(["v1"] * 50)
    assert reader.full_reads == 0


def test_cache_info(image_rows, counting_store):
    store, _ = counting_store(read_mode="auto", full_read_threshold=3)
    empty = store.cache_info()
    assert empty["total_bytes"] == 0
    store.get_region("v1", 0, 5, 0, 5)
    after_region = store.cache_info()
    assert after_region["regions"] == 1
    assert after_region["region_bytes"] == 5 * 5 * 8
    store.get_image("v1")
    after_full = store.cache_info()
    assert after_full["full_images"] == 1
    assert after_full["full_bytes"] == 50 * 50 * 8
    assert after_full["total_bytes"] == after_full["full_bytes"] + after_full["region_bytes"]


def test_cache_info_human(image_rows, counting_store):
    store, _ = counting_store(read_mode="full")
    store.get_image("v1")
    info = store.cache_info(human=True)
    assert info["full_images"] == 1
    assert isinstance(info["full_bytes"], str) and "KB" in info["full_bytes"]
    assert isinstance(info["total_bytes"], str)
