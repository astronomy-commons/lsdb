import os

import hipscat as hc
import pandas as pd
import pytest
from hipscat.pixel_math import hipscat_id_to_healpix

import lsdb
from lsdb.dask.divisions import HIPSCAT_ID_MAX

DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
SMALL_SKY_ORDER1_DIR_NAME = "small_sky_order1"
SMALL_SKY_ORDER1_CSV = "small_sky_order1.csv"
XMATCH_CORRECT_FILE = "xmatch_correct.csv"
XMATCH_CORRECT_005_FILE = "xmatch_correct_0_005.csv"
XMATCH_CORRECT_3N_2T_NO_MARGIN_FILE = "xmatch_correct_3n_2t_no_margin.csv"
XMATCH_MOCK_FILE = "xmatch_mock.csv"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@pytest.fixture
def small_sky_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_DIR_NAME)


@pytest.fixture
def small_sky_xmatch_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_XMATCH_NAME)


@pytest.fixture
def small_sky_order1_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_ORDER1_DIR_NAME)


@pytest.fixture
def small_sky_hipscat_catalog(small_sky_dir):
    return hc.catalog.Catalog.read_from_hipscat(small_sky_dir)


@pytest.fixture
def small_sky_catalog(small_sky_dir):
    return lsdb.read_hipscat(small_sky_dir, catalog_type=lsdb.catalog.Catalog)


@pytest.fixture
def small_sky_xmatch_catalog(small_sky_xmatch_dir):
    return lsdb.read_hipscat(small_sky_xmatch_dir)


@pytest.fixture
def small_sky_order1_hipscat_catalog(small_sky_order1_dir):
    return hc.catalog.Catalog.read_from_hipscat(small_sky_order1_dir)


@pytest.fixture
def small_sky_order1_catalog(small_sky_order1_dir):
    return lsdb.read_hipscat(small_sky_order1_dir)


@pytest.fixture
def small_sky_order1_df(small_sky_order1_dir):
    return pd.read_csv(os.path.join(small_sky_order1_dir, SMALL_SKY_ORDER1_CSV))


@pytest.fixture
def xmatch_correct(small_sky_xmatch_dir):
    return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_FILE))


@pytest.fixture
def xmatch_correct_005(small_sky_xmatch_dir):
    return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_005_FILE))


@pytest.fixture
def xmatch_correct_3n_2t_no_margin(small_sky_xmatch_dir):
    return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_3N_2T_NO_MARGIN_FILE))


@pytest.fixture
def xmatch_mock(small_sky_xmatch_dir):
    return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_MOCK_FILE))

@pytest.fixture
def assert_divisions_are_correct():
    def assert_divisions_are_correct(catalog):
        # Check that number of divisions == number of pixels + 1
        hp_pixels = catalog.get_ordered_healpix_pixels()
        assert len(catalog._ddf.divisions) == len(hp_pixels) + 1
        # Check that the divisions are not None
        assert None not in catalog._ddf.divisions
        # Check that divisions belong to the correct pixel
        for division, hp_pixel in zip(catalog._ddf.divisions, hp_pixels):
            div_pixel = hipscat_id_to_healpix([division], target_order=hp_pixel.order)
            assert hp_pixel.pixel == div_pixel
        # The last division corresponds to the HIPSCAT_ID_MAX
        assert catalog._ddf.divisions[-1] == HIPSCAT_ID_MAX
    return assert_divisions_are_correct
