import os

import hipscat as hc
import pandas as pd
import pytest

import lsdb

DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
SMALL_SKY_NO_METADATA_DIR_NAME = "small_sky_no_metadata"
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
def small_sky_no_metadata_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_NO_METADATA_DIR_NAME)


@pytest.fixture
def small_sky_order1_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_ORDER1_DIR_NAME)


@pytest.fixture
def small_sky_hipscat_catalog(small_sky_dir):
    return hc.catalog.Catalog.read_from_hipscat(small_sky_dir)


@pytest.fixture
def small_sky_catalog(small_sky_dir):
    return lsdb.read_hipscat(small_sky_dir)


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
