import math

import astropy.units as u
import hats as hc
import hats.pixel_math.healpix_shim as hp
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hats.catalog import CatalogType
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from mocpy import MOC

import lsdb
from lsdb.catalog.margin_catalog import MarginCatalog, _validate_margin_catalog


def get_catalog_kwargs(catalog, **kwargs):
    """Generates arguments for a test catalog. By default, the
    partition size is 1 kB, and it is presented in megabytes."""
    catalog_info = catalog.hc_structure.catalog_info
    kwargs = {
        "catalog_name": catalog_info.catalog_name,
        "catalog_type": catalog_info.catalog_type,
        "lowest_order": 0,
        "highest_order": 5,
        "threshold": 50,
        **kwargs,
    }
    return kwargs


def test_from_dataframe(
    small_sky_order1_dir, small_sky_order1_df, small_sky_order1_catalog, assert_divisions_are_correct
):
    """Tests that we can initialize a catalog from a Pandas Dataframe and
    that the loaded content is correct"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    # Catalogs have the same information
    assert (
        catalog.hc_structure.catalog_info.explicit_dict()
        == small_sky_order1_catalog.hc_structure.catalog_info.explicit_dict()
    )
    # Index is set to spatial index
    assert catalog._ddf.index.name == SPATIAL_INDEX_COLUMN
    # Dataframes have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
        small_sky_order1_catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
    )
    # Divisions belong to the respective HEALPix pixels
    assert_divisions_are_correct(catalog)
    # The arrow schema was automatically inferred
    expected_schema = hc.read_hats(small_sky_order1_dir).schema
    assert catalog.hc_structure.schema.equals(expected_schema)

    assert isinstance(catalog.compute(), npd.NestedFrame)


def test_from_dataframe_catalog_of_invalid_type(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that an exception is thrown if the catalog is not of type OBJECT or SOURCE"""
    valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE]
    for catalog_type in CatalogType.all_types():
        kwargs = get_catalog_kwargs(small_sky_order1_catalog, catalog_type=catalog_type)
        if catalog_type in valid_catalog_types:
            lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
        else:
            with pytest.raises(ValueError):
                lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
        # Drop spatial_index that might have been created in place
        small_sky_order1_df.reset_index(drop=True, inplace=True)


def test_from_dataframe_when_threshold_and_partition_size_specified(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that specifying simultaneously threshold and partition_size is invalid"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=10, threshold=10_000)
    with pytest.raises(ValueError, match="Specify only one: threshold or partition_size"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)


def test_partitions_on_map_equal_partitions_in_df(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions on the partition map exist in the Dask Dataframe"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    for hp_pixel, partition_index in catalog._ddf_pixel_map.items():
        partition_df = catalog._ddf.partitions[partition_index].compute()
        assert isinstance(partition_df, pd.DataFrame)
        for _, row in partition_df.iterrows():
            ipix = hp.radec2pix(hp_pixel.order, row["ra"], row["dec"])
            assert ipix == hp_pixel.pixel


def test_partitions_in_partition_info_equal_partitions_on_map(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions in the partition info match those on the partition map"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    for hp_pixel in catalog.hc_structure.get_healpix_pixels():
        partition_from_df = catalog.get_partition(hp_pixel.order, hp_pixel.pixel)
        partition_index = catalog._ddf_pixel_map[hp_pixel]
        partition_from_map = catalog._ddf.partitions[partition_index]
        pd.testing.assert_frame_equal(partition_from_df.compute(), partition_from_map.compute())


def test_partitions_on_map_match_pixel_tree(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that HEALPix pixels on the partition map exist in pixel tree"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    for hp_pixel, _ in catalog._ddf_pixel_map.items():
        assert hp_pixel in catalog.hc_structure.pixel_tree


def test_from_dataframe_with_non_default_ra_dec_columns(small_sky_order1_df, small_sky_order1_catalog):
    """Tests the creation of a catalog using non-default ra and dec columns"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, ra_column="my_ra", dec_column="my_dec")
    # If the columns for ra and dec do not exist
    with pytest.raises(KeyError):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # If they were indeed named differently
    small_sky_order1_df.rename(columns={"ra": "my_ra", "dec": "my_dec"}, inplace=True)
    lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)


def test_partitions_obey_partition_size(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified size"""
    # Use partitions with 10 rows
    partition_size = 10
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=partition_size, threshold=None)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate size of dataframe per partition
    partition_sizes = [len(partition_df) for partition_df in catalog._ddf.partitions]
    assert all(size <= partition_size for size in partition_sizes)


def test_partitions_obey_threshold(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified threshold"""
    threshold = 50
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=None, threshold=threshold)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate number of pixels per partition
    num_partition_pixels = [len(partition_df.compute().index) for partition_df in catalog._ddf.partitions]
    assert all(num_pixels <= threshold for num_pixels in num_partition_pixels)


def test_from_dataframe_large_input(small_sky_order1_catalog, assert_divisions_are_correct):
    """Tests that we can initialize a catalog from a LARGE Pandas Dataframe and
    that we're warned about the catalog's size"""
    original_catalog_info = small_sky_order1_catalog.hc_structure.catalog_info
    kwargs = {
        "catalog_name": original_catalog_info.catalog_name,
        "catalog_type": original_catalog_info.catalog_type,
        "obs_regime": "Optical",
    }

    rng = np.random.default_rng()
    random_df = pd.DataFrame({"ra": rng.uniform(0, 60, 1_500_000), "dec": rng.uniform(0, 60, 1_500_000)})

    # Read CSV file for the small sky order 1 catalog
    with pytest.warns(RuntimeWarning, match="from_dataframe is not intended for large datasets"):
        catalog = lsdb.from_dataframe(random_df, margin_threshold=None, **kwargs)
    assert isinstance(catalog, lsdb.Catalog)
    # Catalogs have the same information
    original_catalog_info.total_rows = 1_500_000
    assert catalog.hc_structure.catalog_info.explicit_dict() == original_catalog_info.explicit_dict()
    assert catalog.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"
    assert catalog.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")
    # Index is set to spatial index
    assert catalog._ddf.index.name == SPATIAL_INDEX_COLUMN
    # Divisions belong to the respective HEALPix pixels
    assert_divisions_are_correct(catalog)


def test_partitions_obey_default_threshold_when_no_arguments_specified(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that partitions are limited by the default threshold
    when no partition size or threshold is specified"""
    df_total_memory = small_sky_order1_df.memory_usage(deep=True).sum()
    partition_memory = df_total_memory / len(small_sky_order1_df)
    default_threshold = math.ceil((1 << 30) / partition_memory)
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, threshold=None, partition_size=None)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate number of pixels per partition
    num_partition_pixels = [len(partition_df.compute().index) for partition_df in catalog._ddf.partitions]
    assert all(num_pixels <= default_threshold for num_pixels in num_partition_pixels)


def test_catalog_pixels_nested_ordering(small_sky_source_df):
    """Tests that the catalog's representation of partitions is ordered by
    nested healpix ordering (breadth-first), instead of numeric by Norder/Npix."""
    catalog = lsdb.from_dataframe(
        small_sky_source_df,
        catalog_name="small_sky_source",
        catalog_type="source",
        lowest_order=0,
        highest_order=2,
        threshold=3_000,
        margin_threshold=None,
        ra_column="source_ra",
        dec_column="source_dec",
    )

    assert len(catalog.get_healpix_pixels()) == 14

    argsort = get_pixel_argsort(catalog.get_healpix_pixels())
    npt.assert_array_equal(argsort, np.arange(0, 14))


def test_from_dataframe_small_sky_source_with_margins(small_sky_source_df, small_sky_source_margin_catalog):
    kwargs = {
        "catalog_name": "small_sky_source",
        "catalog_type": "source",
        "obs_regime": "Optical",
    }
    catalog = lsdb.from_dataframe(
        small_sky_source_df,
        ra_column="source_ra",
        dec_column="source_dec",
        lowest_order=0,
        highest_order=2,
        threshold=3000,
        margin_threshold=180,
        **kwargs,
    )

    assert catalog.margin is not None
    margin = catalog.margin
    assert isinstance(margin, MarginCatalog)
    assert isinstance(margin._ddf, nd.NestedFrame)
    assert margin.get_healpix_pixels() == small_sky_source_margin_catalog.get_healpix_pixels()

    # The points of this margin catalog will be a superset of the hats-imported one,
    # as fine filtering is not enabled here.
    for hp_pixel in margin.hc_structure.get_healpix_pixels():
        partition_from_df = margin.get_partition(hp_pixel.order, hp_pixel.pixel)
        expected_df = small_sky_source_margin_catalog.get_partition(hp_pixel.order, hp_pixel.pixel)
        assert len(expected_df) <= len(partition_from_df)

        margin_source_ids = set(partition_from_df["source_id"])
        expected_source_ids = set(expected_df["source_id"])
        assert len(expected_source_ids - margin_source_ids) == 0

    assert isinstance(margin.compute(), npd.NestedFrame)

    assert catalog.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"
    assert margin.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"

    assert catalog.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")
    assert margin.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")

    # The margin and main catalog's schemas are valid
    _validate_margin_catalog(margin.hc_structure, catalog.hc_structure)


def test_from_dataframe_margin_threshold_from_order(small_sky_source_df):
    # By default, the threshold is set to 5 arcsec, triggering a warning
    with pytest.warns(RuntimeWarning, match="Ignoring margin_threshold"):
        catalog = lsdb.from_dataframe(
            small_sky_source_df,
            ra_column="source_ra",
            dec_column="source_dec",
            lowest_order=0,
            highest_order=2,
            threshold=3000,
            margin_order=3,
        )
    assert len(catalog.margin.get_healpix_pixels()) == 17
    margin_threshold_order3 = hp.order2mindist(3) * 60.0
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == margin_threshold_order3
    assert catalog.margin._ddf.index.name == catalog._ddf.index.name
    _validate_margin_catalog(catalog.margin.hc_structure, catalog.hc_structure)


def test_from_dataframe_invalid_margin_args(small_sky_source_df):
    # The provided margin threshold is negative
    with pytest.raises(ValueError, match="positive"):
        lsdb.from_dataframe(
            small_sky_source_df,
            ra_column="source_ra",
            dec_column="source_dec",
            lowest_order=2,
            margin_threshold=-1,
        )
    # Margin order is inferior to the main catalog's highest order
    with pytest.raises(ValueError, match="margin_order"):
        lsdb.from_dataframe(
            small_sky_source_df,
            ra_column="source_ra",
            dec_column="source_dec",
            lowest_order=2,
            margin_order=1,
        )


def test_from_dataframe_margin_is_empty(small_sky_order1_df):
    catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        highest_order=5,
        threshold=100,
    )
    assert len(catalog.margin.get_healpix_pixels()) == 0
    assert catalog.margin._ddf_pixel_map == {}
    assert catalog.margin._ddf.index.name == catalog._ddf.index.name
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 5.0
    _validate_margin_catalog(catalog.margin.hc_structure, catalog.hc_structure)


def test_from_dataframe_margin_threshold_zero(small_sky_order1_df):
    catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        highest_order=5,
        threshold=100,
        margin_threshold=0,
    )
    assert len(catalog.margin.get_healpix_pixels()) == 0
    assert catalog.margin._ddf_pixel_map == {}
    assert catalog.margin._ddf.index.name == catalog._ddf.index.name
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 0
    _validate_margin_catalog(catalog.margin.hc_structure, catalog.hc_structure)


def test_from_dataframe_moc(small_sky_order1_catalog):
    order = 1
    pixels = [44, 45, 46]
    partitions = [small_sky_order1_catalog.get_partition(order, p).compute() for p in pixels]
    df = pd.concat(partitions)
    subset_catalog = lsdb.from_dataframe(df)
    assert subset_catalog.hc_structure.moc is not None
    assert np.all(subset_catalog.hc_structure.moc.degrade_to_order(1).flatten() == pixels)
    correct_moc = MOC.from_lonlat(
        lon=df["ra"].to_numpy() * u.deg, lat=df["dec"].to_numpy() * u.deg, max_norder=10
    )
    assert correct_moc == subset_catalog.hc_structure.moc


def test_from_dataframe_moc_params(small_sky_order1_catalog):
    order = 1
    pixels = [44, 45, 46]
    max_order = 5
    partitions = [small_sky_order1_catalog.get_partition(order, p).compute() for p in pixels]
    df = pd.concat(partitions)
    subset_catalog = lsdb.from_dataframe(df, moc_max_order=max_order)
    assert subset_catalog.hc_structure.moc is not None
    assert subset_catalog.hc_structure.moc.max_order == max_order
    assert np.all(subset_catalog.hc_structure.moc.degrade_to_order(1).flatten() == pixels)
    correct_moc = MOC.from_lonlat(
        lon=df["ra"].to_numpy() * u.deg, lat=df["dec"].to_numpy() * u.deg, max_norder=max_order
    )
    assert correct_moc == subset_catalog.hc_structure.moc


def test_from_dataframe_without_moc(small_sky_order1_catalog):
    order = 1
    pixels = [44, 45, 46]
    max_order = 5
    partitions = [small_sky_order1_catalog.get_partition(order, p).compute() for p in pixels]
    df = pd.concat(partitions)
    subset_catalog = lsdb.from_dataframe(df, moc_max_order=max_order, should_generate_moc=False)
    assert subset_catalog.hc_structure.moc is None


def test_from_dataframe_with_backend(small_sky_order1_df, small_sky_order1_dir):
    """Tests that we can initialize a catalog from a Pandas Dataframe with the desired backend"""
    # Read the catalog from hats format using pyarrow, import it from a CSV using
    # the same backend and assert that we obtain the same catalog
    expected_catalog = lsdb.read_hats(small_sky_order1_dir)
    kwargs = get_catalog_kwargs(expected_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    assert all(isinstance(col_type, pd.ArrowDtype) for col_type in catalog.dtypes)
    pd.testing.assert_frame_equal(
        catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
        expected_catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
    )

    # Test that we can also keep the original types if desired
    expected_catalog = lsdb.read_hats(small_sky_order1_dir, dtype_backend=None)
    kwargs = get_catalog_kwargs(expected_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, use_pyarrow_types=False, **kwargs)
    assert all(isinstance(col_type, np.dtype) for col_type in catalog.dtypes)
    pd.testing.assert_frame_equal(
        catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
        expected_catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
    )


def test_from_dataframe_with_arrow_schema(small_sky_order1_df, small_sky_order1_dir):
    expected_schema = hc.read_hats(small_sky_order1_dir).schema
    catalog = lsdb.from_dataframe(small_sky_order1_df, schema=expected_schema)
    assert catalog.hc_structure.schema is expected_schema
