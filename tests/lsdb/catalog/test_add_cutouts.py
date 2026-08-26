import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import lsdb
from lsdb import Catalog
from lsdb.cutouts import CatalogImageStore, ChainImageStore, CutoutSeries

IMAGE_SIZE = 200


def make_wcs(ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [IMAGE_SIZE / 2, IMAGE_SIZE / 2]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture(scope="module")
def image_catalog(tmp_path_factory):
    """Two 2x2 deg images with constant pixel values 100 and 101."""
    image_dir = tmp_path_factory.mktemp("images")
    specs, wcs_list = [], []
    for position, ra in enumerate([180.0, 181.5]):
        path = str(image_dir / f"v{position}.fits")
        fits.PrimaryHDU(data=np.full((IMAGE_SIZE, IMAGE_SIZE), 100.0 + position)).writeto(path)
        specs.append({"image_id": f"v{position}", "path": path, "width": IMAGE_SIZE, "height": IMAGE_SIZE})
        wcs_list.append(make_wcs(ra, -30.0))
    return lsdb.from_images(
        pd.DataFrame(specs), wcs_list, threshold=100, catalog_name="imgs", image_format="fits"
    )


@pytest.fixture(scope="module")
def object_catalog():
    """40 objects in the image region, 20 far outside all coverage."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "id": range(60),
            "ra": np.concatenate([rng.uniform(179.2, 182.3, 40), rng.uniform(200, 210, 20)]),
            "dec": np.concatenate([rng.uniform(-30.8, -29.2, 40), rng.uniform(-10, 0, 20)]),
        }
    )
    return lsdb.from_dataframe(frame, catalog_name="objs")


def test_add_cutouts_is_lazy(object_catalog, image_catalog):
    lazy = object_catalog.add_cutouts(image_catalog, stamp_size=15)
    assert isinstance(lazy, Catalog)
    assert "cutouts" in lazy.columns
    assert str(lazy.dtypes["cutouts"]) == "cutout"
    # Original catalog is untouched
    assert "cutouts" not in object_catalog.columns


def test_add_cutouts_left_semantics(object_catalog, image_catalog):
    result = object_catalog.add_cutouts(image_catalog, stamp_size=15).compute()
    # Every object is kept exactly once, NA where uncovered
    assert sorted(result["id"]) == list(range(60))
    covered = result["cutouts"].notna()
    assert covered.sum() > 0
    assert set(result.loc[~covered, "id"]) >= set(range(40, 60))


def test_add_cutouts_renders_through_store(object_catalog, image_catalog):
    result = object_catalog.add_cutouts(image_catalog, stamp_size=15).compute()
    series = result["cutouts"]
    assert type(series) is CutoutSeries
    # Partition stores merge on concat: a single store or a chain of them
    assert isinstance(series.store, CatalogImageStore | ChainImageStore)
    matched = CutoutSeries(series.dropna())
    image_ids = matched.descriptors()["image_id"]
    for data, image_id in zip(matched.head(5).array, image_ids.head(5), strict=True):
        assert data.shape == (15, 15)
        # Constant image values identify the source image
        expected = 100.0 if image_id == "v0" else 101.0
        assert float(data[0, 0]) == expected


def test_add_cutouts_splits_to_finer_image_tree(object_catalog, tmp_path):
    # threshold=1 forces the image tree deeper than the object tree
    specs, wcs_list = [], []
    for position, ra in enumerate([180.0, 180.5]):
        path = str(tmp_path / f"deep{position}.fits")
        fits.PrimaryHDU(data=np.zeros((IMAGE_SIZE, IMAGE_SIZE))).writeto(path)
        specs.append({"image_id": f"deep{position}", "path": path, "width": IMAGE_SIZE, "height": IMAGE_SIZE})
        wcs_list.append(make_wcs(ra, -30.0))
    with pytest.warns(UserWarning):
        deep_images = lsdb.from_images(pd.DataFrame(specs), wcs_list, threshold=1)
    result_catalog = object_catalog.add_cutouts(deep_images, stamp_size=5)
    object_orders = {pixel.order for pixel in object_catalog.get_healpix_pixels()}
    result_orders = {pixel.order for pixel in result_catalog.get_healpix_pixels()}
    assert max(result_orders) > max(object_orders)
    # No rows lost or duplicated by the split
    result = result_catalog.compute()
    assert sorted(result["id"]) == list(range(60))


def test_add_cutouts_stamp_changes_matches(object_catalog, image_catalog):
    small = object_catalog.add_cutouts(image_catalog, stamp_size=5).compute()
    huge = object_catalog.add_cutouts(image_catalog, stamp_size=190).compute()
    # A stamp nearly the size of the image fits almost nowhere
    assert huge["cutouts"].notna().sum() < small["cutouts"].notna().sum()


def test_add_cutouts_column_name(object_catalog, image_catalog):
    result = object_catalog.add_cutouts(image_catalog, stamp_size=15, column_name="stamps").compute()
    assert "stamps" in result.columns
    with pytest.raises(ValueError, match="already exists"):
        object_catalog.add_cutouts(image_catalog, stamp_size=15, column_name="ra")


def test_add_cutouts_descriptors_only(object_catalog, image_catalog):
    result = object_catalog.add_cutouts(image_catalog, stamp_size=15, attach_store=False).compute()
    series = result["cutouts"]
    matched = series.dropna()
    assert len(matched) > 0
    assert series.store is None
    with pytest.raises(ValueError, match="no image store"):
        _ = matched.iloc[0]


def test_add_cutouts_query_composes(object_catalog, image_catalog):
    # The cutout column survives further lazy operations
    lazy = object_catalog.add_cutouts(image_catalog, stamp_size=15).query("id < 30")
    result = lazy.compute()
    assert len(result) == 30
    assert result["cutouts"].notna().sum() > 0
    assert result["cutouts"].dropna().iloc[0].shape == (15, 15)
