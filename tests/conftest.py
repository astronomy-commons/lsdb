from pathlib import Path

import hats as hc
import pandas as pd
import pytest
from hats.pixel_math import spatial_index_to_healpix
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, healpix_to_spatial_index

import lsdb

DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_LEFT_XMATCH_NAME = "small_sky_left_xmatch"
SMALL_SKY_SOURCE_DIR_NAME = "small_sky_source"
SMALL_SKY_SOURCE_MARGIN_NAME = "small_sky_source_margin"
SMALL_SKY_ORDER3_SOURCE_MARGIN_NAME = "small_sky_order3_source_margin"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
SMALL_SKY_XMATCH_MARGIN_NAME = "small_sky_xmatch_margin"
SMALL_SKY_TO_XMATCH_NAME = "small_sky_to_xmatch"
SMALL_SKY_TO_XMATCH_SOFT_NAME = "small_sky_to_xmatch_soft"
SMALL_SKY_ORDER1_DIR_NAME = "small_sky_order1"
SMALL_SKY_ORDER1_NO_PANDAS_DIR_NAME = "small_sky_order1_no_pandas_meta"
SMALL_SKY_ORDER1_SOURCE_NAME = "small_sky_order1_source"
SMALL_SKY_ORDER1_SOURCE_MARGIN_NAME = "small_sky_order1_source_margin"
SMALL_SKY_TO_ORDER1_SOURCE_NAME = "small_sky_to_o1source"
SMALL_SKY_TO_ORDER1_SOURCE_SOFT_NAME = "small_sky_to_o1source_soft"
SMALL_SKY_NO_METADATA = "small_sky_no_metadata"
XMATCH_CORRECT_FILE = "xmatch_correct.csv"
XMATCH_CORRECT_005_FILE = "xmatch_correct_0_005.csv"
XMATCH_CORRECT_002_005_FILE = "xmatch_correct_002_005.csv"
XMATCH_CORRECT_05_2_3N_MARGIN_FILE = "xmatch_correct_05_2_3n_margin.csv"
XMATCH_CORRECT_3N_2T_FILE = "xmatch_correct_3n_2t.csv"
XMATCH_CORRECT_3N_2T_NO_MARGIN_FILE = "xmatch_correct_3n_2t_no_margin.csv"
XMATCH_CORRECT_3N_2T_NEGATIVE_FILE = "xmatch_correct_3n_2t_negative.csv"
XMATCH_MOCK_FILE = "xmatch_mock.csv"
TEST_DIR = Path(__file__).parent


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / DATA_DIR_NAME


@pytest.fixture
def small_sky_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_DIR_NAME


@pytest.fixture
def small_sky_left_xmatch_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_LEFT_XMATCH_NAME


@pytest.fixture
def small_sky_xmatch_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_XMATCH_NAME


@pytest.fixture
def small_sky_xmatch_margin_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_XMATCH_MARGIN_NAME


@pytest.fixture
def small_sky_to_xmatch_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_TO_XMATCH_NAME


@pytest.fixture
def small_sky_to_xmatch_soft_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_TO_XMATCH_SOFT_NAME


@pytest.fixture
def small_sky_order1_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_ORDER1_DIR_NAME


@pytest.fixture
def small_sky_order1_no_pandas_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_ORDER1_NO_PANDAS_DIR_NAME


@pytest.fixture
def small_sky_order1_source_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_ORDER1_SOURCE_NAME


@pytest.fixture
def small_sky_source_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_SOURCE_DIR_NAME


@pytest.fixture
def small_sky_source_catalog(small_sky_source_dir):
    return lsdb.read_hats(small_sky_source_dir)


@pytest.fixture
def small_sky_order1_source_margin_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_ORDER1_SOURCE_MARGIN_NAME


@pytest.fixture
def small_sky_to_order1_source_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_TO_ORDER1_SOURCE_NAME


@pytest.fixture
def small_sky_to_order1_source_soft_dir(test_data_dir):
    return test_data_dir / SMALL_SKY_TO_ORDER1_SOURCE_SOFT_NAME


@pytest.fixture
def small_sky_hats_catalog(small_sky_dir):
    return hc.read_hats(small_sky_dir)


@pytest.fixture
def small_sky_order1_id_index_dir(test_data_dir):
    return test_data_dir / "small_sky_order1_id_index"


@pytest.fixture
def small_sky_catalog(small_sky_dir):
    return lsdb.read_hats(small_sky_dir)


@pytest.fixture
def small_sky_left_xmatch_catalog(small_sky_left_xmatch_dir):
    return lsdb.read_hats(small_sky_left_xmatch_dir)


@pytest.fixture
def small_sky_xmatch_catalog(small_sky_xmatch_dir):
    return lsdb.read_hats(small_sky_xmatch_dir)


@pytest.fixture
def small_sky_xmatch_margin_catalog(small_sky_xmatch_margin_dir):
    return lsdb.read_hats(small_sky_xmatch_margin_dir)


@pytest.fixture
def small_sky_xmatch_with_margin(small_sky_xmatch_dir, small_sky_xmatch_margin_dir):
    return lsdb.read_hats(small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir)


@pytest.fixture
def small_sky_to_xmatch_catalog(small_sky_to_xmatch_dir):
    return lsdb.read_hats(small_sky_to_xmatch_dir)


@pytest.fixture
def small_sky_to_xmatch_soft_catalog(small_sky_to_xmatch_soft_dir):
    return lsdb.read_hats(small_sky_to_xmatch_soft_dir)


@pytest.fixture
def small_sky_order1_hats_catalog(small_sky_order1_dir):
    return hc.read_hats(small_sky_order1_dir)


@pytest.fixture
def small_sky_order1_catalog(small_sky_order1_dir):
    return lsdb.read_hats(small_sky_order1_dir)


@pytest.fixture
def small_sky_order1_source_with_margin(small_sky_order1_source_dir, small_sky_order1_source_margin_dir):
    return lsdb.read_hats(small_sky_order1_source_dir, margin_cache=small_sky_order1_source_margin_dir)


@pytest.fixture
def small_sky_order1_source_margin_catalog(small_sky_order1_source_margin_dir):
    return lsdb.read_hats(small_sky_order1_source_margin_dir)


@pytest.fixture
def small_sky_to_o1source_catalog(small_sky_to_order1_source_dir):
    return lsdb.read_hats(small_sky_to_order1_source_dir)


@pytest.fixture
def small_sky_to_o1source_soft_catalog(small_sky_to_order1_source_soft_dir):
    return lsdb.read_hats(small_sky_to_order1_source_soft_dir)


@pytest.fixture
def small_sky_order1_df(test_data_dir):
    return pd.read_csv(test_data_dir / "raw" / "small_sky" / "small_sky.csv")


@pytest.fixture
def small_sky_source_df(test_data_dir):
    return pd.read_csv(test_data_dir / "raw" / "small_sky_source" / "small_sky_source.csv")


@pytest.fixture
def small_sky_source_margin_catalog(test_data_dir):
    return lsdb.read_hats(test_data_dir / SMALL_SKY_SOURCE_MARGIN_NAME)


@pytest.fixture
def small_sky_order3_source_margin_catalog(test_data_dir):
    return lsdb.read_hats(test_data_dir / SMALL_SKY_ORDER3_SOURCE_MARGIN_NAME)


@pytest.fixture
def small_sky_with_nested_sources(small_sky_order1_catalog, small_sky_order1_source_with_margin):
    return small_sky_order1_catalog.join_nested(
        small_sky_order1_source_with_margin, left_on="id", right_on="object_id", nested_column_name="sources"
    )


@pytest.fixture
def small_sky_no_metadata_dir(test_data_dir):
    return test_data_dir / "raw" / SMALL_SKY_NO_METADATA


@pytest.fixture
def xmatch_expected_dir(test_data_dir):
    return test_data_dir / "raw" / "xmatch_expected"


@pytest.fixture
def xmatch_correct(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_FILE)


@pytest.fixture
def xmatch_correct_005(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_005_FILE)


@pytest.fixture
def xmatch_correct_002_005(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_002_005_FILE)


@pytest.fixture
def xmatch_correct_05_2_3n_margin(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_05_2_3N_MARGIN_FILE)


@pytest.fixture
def xmatch_correct_3n_2t(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_3N_2T_FILE)


@pytest.fixture
def xmatch_correct_3n_2t_no_margin(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_3N_2T_NO_MARGIN_FILE)


@pytest.fixture
def xmatch_correct_3n_2t_negative(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_CORRECT_3N_2T_NEGATIVE_FILE)


@pytest.fixture
def xmatch_mock(xmatch_expected_dir):
    return pd.read_csv(xmatch_expected_dir / XMATCH_MOCK_FILE)


@pytest.fixture
def cone_search_expected_dir(test_data_dir):
    return test_data_dir / "raw" / "cone_search_expected"


@pytest.fixture
def cone_search_expected(cone_search_expected_dir):
    return pd.read_csv(cone_search_expected_dir / "catalog.csv", index_col=SPATIAL_INDEX_COLUMN)


@pytest.fixture
def cone_search_margin_expected(cone_search_expected_dir):
    return pd.read_csv(cone_search_expected_dir / "margin.csv", index_col=SPATIAL_INDEX_COLUMN)


@pytest.fixture
def assert_divisions_are_correct():
    def assert_divisions_are_correct(catalog):
        # Check that number of divisions == number of pixels + 1
        hp_pixels = [None] * len(catalog._ddf_pixel_map)
        for pix, index in catalog._ddf_pixel_map.items():
            hp_pixels[index] = pix
        if len(hp_pixels) == 0:
            # Special case if there are no partitions.
            assert catalog._ddf.divisions == (None, None)
            return
        assert len(catalog._ddf.divisions) == len(hp_pixels) + 1
        # Check that the divisions are not None
        assert None not in catalog._ddf.divisions
        # Check that divisions belong to the correct pixel
        for division, hp_pixel in zip(catalog._ddf.divisions, hp_pixels):
            div_pixel = spatial_index_to_healpix([division], target_order=hp_pixel.order)
            assert hp_pixel.pixel == div_pixel
        # The last division corresponds to the largest healpix value
        assert catalog._ddf.divisions[-1] == healpix_to_spatial_index(
            hp_pixels[-1].order, hp_pixels[-1].pixel + 1
        )

    return assert_divisions_are_correct
