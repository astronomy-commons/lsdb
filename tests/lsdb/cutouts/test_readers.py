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
    array = CutoutArray.from_arrays(["v1", "v1"], x0=[0, 10], y0=[0, 10], width=[5, 5], height=[5, 5])
    series = pd.Series(array).cutout.with_store(store)
    np.testing.assert_array_equal(series.iloc[1].data, data[10:15, 10:15])
    cutout2d = series.iloc[1].to_cutout2d()
    assert cutout2d.wcs is not None
