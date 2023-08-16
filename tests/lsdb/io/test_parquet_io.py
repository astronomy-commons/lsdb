import hipscat as hc
import pandas as pd
import pyarrow.parquet as pq

from lsdb.io import read_parquet_file_to_pandas, read_parquet_schema


def test_read_parquet_data(small_sky_order1_hipscat_catalog):
    pixel = small_sky_order1_hipscat_catalog.partition_info.get_healpix_pixels()[0]
    file = hc.io.paths.pixel_catalog_file(
        small_sky_order1_hipscat_catalog.catalog_base_dir, pixel.order, pixel.pixel
    )
    file_pointer = hc.io.get_file_pointer_from_path(file)
    dataframe = read_parquet_file_to_pandas(file_pointer)
    loaded_df = pd.read_parquet(file)
    pd.testing.assert_frame_equal(dataframe, loaded_df)


def test_read_parquet_schema(small_sky_order1_hipscat_catalog):
    pixel = small_sky_order1_hipscat_catalog.partition_info.get_healpix_pixels()[0]
    file = hc.io.paths.pixel_catalog_file(
        small_sky_order1_hipscat_catalog.catalog_base_dir, pixel.order, pixel.pixel
    )
    file_pointer = hc.io.get_file_pointer_from_path(file)
    schema = read_parquet_schema(file_pointer)
    loaded_schema = pq.read_schema(file)
    assert schema.equals(loaded_schema)
