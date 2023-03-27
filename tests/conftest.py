import os

import hipscat as hc
import pytest

import lsdb

DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_ORDER1_DIR_NAME = "small_sky_order1"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@pytest.fixture
def small_sky_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_DIR_NAME)


@pytest.fixture
def small_sky_order1_dir(test_data_dir):
    return os.path.join(test_data_dir, SMALL_SKY_ORDER1_DIR_NAME)


@pytest.fixture
def small_sky_hipscat_catalog(small_sky_dir):
    return hc.catalog.Catalog(small_sky_dir)


@pytest.fixture
def small_sky_catalog(small_sky_dir):
    return lsdb.read_hipscat(small_sky_dir)


@pytest.fixture
def small_sky_order1_hipscat_catalog(small_sky_order1_dir):
    return hc.catalog.Catalog(small_sky_order1_dir)


@pytest.fixture
def small_sky_order1_catalog(small_sky_order1_dir):
    return lsdb.read_hipscat(small_sky_order1_dir)
