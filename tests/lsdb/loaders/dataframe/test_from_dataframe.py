import math
from importlib.metadata import version

import astropy.units as u
import hats as hc
import hats.pixel_math.healpix_shim as hp
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
import lsdb.nested as nd
from lsdb.catalog.margin_catalog import MarginCatalog, _validate_margin_catalog
from lsdb.operations.operation import Operation


def get_catalog_kwargs(catalog, **kwargs):
    """Generates arguments for a test catalog. By default, the
    partition size is 1 kB, and it is presented in megabytes."""
    catalog_info = catalog.hc_structure.catalog_info
    kwargs = {
        "catalog_name": catalog_info.catalog_name,
        "catalog_type": catalog_info.catalog_type,
        "lowest_order": 0,
        "highest_order": 5,
        "partition_rows": 50,
        **kwargs,
    }
    return kwargs


def test_from_dataframe(small_sky_order1_df, small_sky_order1_catalog, helpers):
    """Tests that we can initialize a catalog from a Pandas Dataframe and
    that the loaded content is correct"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog.meta, npd.NestedFrame)
    # Catalogs have the same information
    # New catalog doesn't have a skymap order yet.
    helpers.assert_catalog_info_is_correct(
        small_sky_order1_catalog.hc_structure.catalog_info,
        catalog.hc_structure.catalog_info,
        do_not_compare=["skymap_order", "moc_sky_fraction", "hats_max_rows"],
        check_extra_properties=False,
    )
    # The hats builder property is set correctly
    assert (
        catalog.hc_structure.catalog_info.hats_builder == f"lsdb v{version('lsdb')}, hats v{version('hats')}"
    )
    # The partition_rows threshold was specified properly
    assert catalog.hc_structure.catalog_info.hats_max_rows <= 50
    # Index is set to spatial index
    assert catalog.meta.index.name == SPATIAL_INDEX_COLUMN
    # Dataframes have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
        small_sky_order1_catalog.compute().sort_values([SPATIAL_INDEX_COLUMN, "id"]),
    )
    # The arrow schema was automatically inferred
    helpers.assert_schema_correct(catalog)
    assert isinstance(catalog.compute(), npd.NestedFrame)
    assert catalog.hc_structure.snapshot is not None
    assert catalog.hc_structure.original_schema == catalog.hc_structure.schema


def test_from_dataframe_catalog_of_invalid_type(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that an exception is thrown if the catalog is not of type OBJECT or SOURCE"""
    valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE, CatalogType.MAP]
    for catalog_type in CatalogType.all_types():
        kwargs = get_catalog_kwargs(small_sky_order1_catalog, catalog_type=catalog_type)
        if catalog_type in valid_catalog_types:
            lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
        else:
            with pytest.raises(ValueError, match="Cannot create"):
                lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
        # Drop spatial_index that might have been created in place
        small_sky_order1_df.reset_index(drop=True, inplace=True)


def test_from_dataframe_invalid_partitioning_parameters(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that deprecated and conflicting partitioning parameters raise exceptions"""
    # Fail when specifying both partition_rows and partition_bytes
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_rows=10, partition_bytes=10_000)
    with pytest.raises(ValueError, match="Specify only one:"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)

    # Fail when using deprecated threshold parameter
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, threshold=10_000)
    with pytest.raises(ValueError, match="is deprecated"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)

    # Fail when user specifies hats_max_rows or hats_max_bytes in kwargs
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_rows=10, hats_max_rows=10)
    with pytest.raises(ValueError, match="hats_max_rows should not be provided in kwargs"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    kwargs = get_catalog_kwargs(
        small_sky_order1_catalog, partition_rows=None, partition_bytes=10_000, hats_max_bytes=10_000
    )
    with pytest.raises(ValueError, match="hats_max_bytes should not be provided in kwargs"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)


def test_partitions_on_healpix_pixels_equal_partitions_in_df(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions on the partition map exist in the Dask Dataframe"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    for partition_index, hp_pixel in enumerate(catalog.get_healpix_pixels()):
        partition_df = catalog.partitions[partition_index].compute()
        assert isinstance(partition_df, npd.NestedFrame)
        for _, row in partition_df.iterrows():
            ipix = hp.radec2pix(hp_pixel.order, row["ra"], row["dec"])
            assert ipix == hp_pixel.pixel


def test_partitions_in_partition_info_equal_partitions_on_operation(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that partitions in the partition info match those on the partition map"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    assert catalog.get_healpix_pixels() == catalog._operation.healpix_pixels


def test_partitions_on_operation_match_pixel_tree(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that HEALPix pixels on the partition map exist in pixel tree"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    for hp_pixel in catalog._operation.healpix_pixels:
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


def test_partitions_obey_partition_rows(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified size"""
    # Use partitions with 10 rows
    partition_rows = 10
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(
        small_sky_order1_catalog,
        partition_rows=partition_rows,
        partition_bytes=None,
    )
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate size of dataframe per partition
    rows_per_partition = [len(partition.compute()) for partition in catalog.partitions]
    assert all(size <= partition_rows for size in rows_per_partition)


def test_partitions_obey_partition_bytes(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified size in bytes"""
    # Use partitions with approximately 1 kB
    partition_bytes = 1 << 10  # 1 kB
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(
        small_sky_order1_catalog,
        partition_rows=None,
        partition_bytes=partition_bytes,
    )
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate size of dataframe per partition
    for partition in catalog.partitions:
        partition_memory = partition.compute().memory_usage(deep=True).sum()
        assert partition_memory <= partition_bytes


def test_from_dataframe_large_input(small_sky_order1_catalog, helpers):
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
        catalog = lsdb.from_dataframe(
            random_df,
            margin_threshold=None,
            partition_rows=int(original_catalog_info.hats_max_rows),
            **kwargs,
        )
    assert isinstance(catalog, lsdb.Catalog)
    # Catalogs have the same information
    original_catalog_info.total_rows = 1_500_000
    # New catalog doesn't have a skymap order or moc_sky_fraction yet.
    expected_dict = {
        k: v
        for k, v in original_catalog_info.explicit_dict().items()
        if k not in ["skymap_order", "moc_sky_fraction", "hats_estsize"]
    }
    test_dict = catalog.hc_structure.catalog_info.explicit_dict()
    test_dict.pop("hats_estsize", None)
    assert test_dict == expected_dict
    assert catalog.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"
    assert catalog.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")
    # Index is set to spatial index
    assert catalog.meta.index.name == SPATIAL_INDEX_COLUMN


def test_partitions_obey_default_partition_rows_when_no_arguments_specified(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that partitions are limited by the default partition size
    when no partition_rows or partition_bytes is specified"""
    df_total_memory = small_sky_order1_df.memory_usage(deep=True).sum()
    partition_memory = df_total_memory / len(small_sky_order1_df)
    default_threshold = math.ceil((1 << 30) / partition_memory)
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_rows=None, partition_bytes=None)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None, **kwargs)
    # Calculate number of pixels per partition
    num_partition_pixels = [len(partition.compute().index) for partition in catalog.partitions]
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
        partition_rows=3_000,
        margin_threshold=None,
        ra_column="source_ra",
        dec_column="source_dec",
    )

    assert len(catalog.get_healpix_pixels()) == 14

    argsort = get_pixel_argsort(catalog.get_healpix_pixels())
    npt.assert_array_equal(argsort, np.arange(0, 14))


def test_from_dataframe_small_sky_source_with_margins(
    small_sky_source_df, small_sky_source_margin_catalog, helpers
):
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
        partition_rows=3000,
        margin_threshold=180,
        margin_order=8,
        **kwargs,
    )

    assert catalog.margin is not None
    margin = catalog.margin
    assert isinstance(margin, MarginCatalog)
    assert isinstance(margin.meta, npd.NestedFrame)
    assert margin.get_healpix_pixels() == small_sky_source_margin_catalog.get_healpix_pixels()

    # The points of this margin catalog will be a superset of the hats-imported one,
    # as fine filtering is not enabled here.
    for hp_pixel in margin.hc_structure.get_healpix_pixels():
        partition_from_cat = margin.get_partition(hp_pixel.order, hp_pixel.pixel)
        expected_cat = small_sky_source_margin_catalog.get_partition(hp_pixel.order, hp_pixel.pixel)
        assert len(expected_cat.compute()) <= len(partition_from_cat.compute())

        margin_source_ids = set(partition_from_cat.compute()["source_id"])
        expected_source_ids = set(expected_cat.compute()["source_id"])
        assert len(expected_source_ids - margin_source_ids) == 0

    assert isinstance(margin.compute(), npd.NestedFrame)

    assert catalog.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"
    assert margin.hc_structure.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"

    assert catalog.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")
    assert margin.hc_structure.catalog_info.__pydantic_extra__["hats_builder"].startswith("lsdb")

    # The margin and main catalog's schemas are valid
    _validate_margin_catalog(margin, catalog)
    helpers.assert_schema_correct(margin)
    helpers.assert_schema_correct(catalog)


def test_from_dataframe_margin_threshold_from_order(small_sky_source_df, helpers):
    # By default, the margin threshold is set to 5 arcsec, triggering a warning
    with pytest.warns(RuntimeWarning, match="Ignoring margin_threshold"):
        catalog = lsdb.from_dataframe(
            small_sky_source_df,
            ra_column="source_ra",
            dec_column="source_dec",
            lowest_order=0,
            highest_order=2,
            partition_rows=3000,
            margin_order=3,
        )
    assert len(catalog.margin.get_healpix_pixels()) == 19
    margin_threshold_order3 = hp.order2mindist(3) * 60.0
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == margin_threshold_order3
    assert catalog.margin.meta.index.name == catalog.meta.index.name
    _validate_margin_catalog(catalog.margin, catalog)
    helpers.assert_schema_correct(catalog.margin)
    helpers.assert_schema_correct(catalog)


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


def test_from_dataframe_margin_is_empty(small_sky_order1_df, helpers):
    catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        highest_order=5,
        partition_rows=100,
    )
    assert len(catalog.margin.get_healpix_pixels()) == 0
    assert catalog.margin._operation.healpix_pixels == []
    assert catalog.margin.meta.index.name == catalog.meta.index.name
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 5.0
    _validate_margin_catalog(catalog.margin, catalog)
    helpers.assert_schema_correct(catalog.margin)
    helpers.assert_schema_correct(catalog)


def test_from_dataframe_margin_threshold_zero(small_sky_order1_df, helpers):
    catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        highest_order=5,
        partition_rows=100,
        margin_threshold=0,
    )
    assert len(catalog.margin.get_healpix_pixels()) == 0
    assert catalog.margin._operation.healpix_pixels == []
    assert catalog.margin.meta.index.name == catalog.meta.index.name
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 0
    _validate_margin_catalog(catalog.margin, catalog)
    helpers.assert_schema_correct(catalog.margin)
    helpers.assert_schema_correct(catalog)


def test_from_dataframe_moc(small_sky_order1_catalog):
    order = 1
    pixels = [44, 45, 46]
    partitions = [small_sky_order1_catalog.get_partition(order, p).compute() for p in pixels]
    df = pd.concat(partitions)
    subset_catalog = lsdb.from_dataframe(df, margin_threshold=None)
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
    subset_catalog = lsdb.from_dataframe(df, moc_max_order=max_order, margin_threshold=None)
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
    subset_catalog = lsdb.from_dataframe(
        df, moc_max_order=max_order, should_generate_moc=False, margin_threshold=None
    )
    assert subset_catalog.hc_structure.moc is None


def test_from_dataframe_with_arrow_schema(small_sky_order1_df, small_sky_order1_dir):
    expected_schema = hc.read_hats(small_sky_order1_dir).schema
    catalog = lsdb.from_dataframe(small_sky_order1_df, schema=expected_schema, margin_threshold=None)
    assert catalog.hc_structure.schema is expected_schema


def test_from_dataframe_keeps_named_index(small_sky_order1_df):
    assert small_sky_order1_df.index.name is None
    small_sky_order1_df.set_index("id", inplace=True)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None)
    assert catalog.meta.index.name == "_healpix_29"
    assert "id" in catalog.columns
    ids = catalog.compute()["id"].to_numpy()
    expected_ids = small_sky_order1_df.index.to_numpy()
    assert np.array_equal(ids, expected_ids)


def test_from_dataframe_does_not_keep_unnamed_index(small_sky_order1_df):
    assert small_sky_order1_df.index.name is None
    range_index = pd.RangeIndex(start=0, stop=len(small_sky_order1_df), step=1)
    assert small_sky_order1_df.index.equals(range_index)
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None)
    assert catalog.meta.index.name == "_healpix_29"
    assert "index" not in catalog.columns


def test_from_dataframe_all_sky(sm_all_sky_df):
    """Regression test for numpy error seen in issue #718

    TypeError: cannot unpack non-iterable numpy.int32 object
    """
    catalog = lsdb.from_dataframe(
        sm_all_sky_df, ra_column="RA", dec_column="DEC", drop_empty_siblings=True, margin_threshold=None
    )
    assert catalog.meta.index.name == "_healpix_29"
    assert len(catalog.get_healpix_pixels()) == 12


def test_from_dataframe_finds_radec_columns(small_sky_order1_df):
    """Check that the RA and Dec columns are identified
    case-insensitively when omitted by the user."""
    # The columns are named "ra" and "dec"
    catalog = lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None)
    assert {"ra", "dec"}.issubset(catalog.columns)
    # It would also work if they were named "Ra" and "Dec"
    df_renamed = small_sky_order1_df.rename(columns={"ra": "Ra", "dec": "Dec"})
    catalog = lsdb.from_dataframe(df_renamed, margin_threshold=None)
    assert {"Ra", "Dec"}.issubset(catalog.columns)
    # If no matches are found, an error is raised
    df_no_radec = small_sky_order1_df.drop(columns=["ra", "dec"])
    with pytest.raises(ValueError, match="No column found"):
        lsdb.from_dataframe(df_no_radec, margin_threshold=None)
    # If multiple matches are found it's ambiguous, and an error is raised
    small_sky_order1_df["RA"] = small_sky_order1_df["ra"].copy()
    with pytest.raises(ValueError, match="possible columns"):
        lsdb.from_dataframe(small_sky_order1_df, margin_threshold=None)


def test_from_dataframe_with_nan_radec():
    """Test that from_dataframe raises a helpful error when NaN values are present in RA/Dec columns."""
    df = pd.DataFrame({"ra": [10.0, np.nan, 30.0], "dec": [20.0, 40.0, np.nan], "id": [1, 2, 3]})
    # Should raise ValueError with a helpful message
    with pytest.raises(ValueError, match=r"NaN values found in .+ columns"):
        lsdb.from_dataframe(df, margin_threshold=None)

    # Also test with custom column names
    df2 = df.rename(columns={"ra": "my_ra", "dec": "my_dec"})
    with pytest.raises(ValueError, match=r"NaN values found in .+ columns"):
        lsdb.from_dataframe(df2, ra_column="my_ra", dec_column="my_dec", margin_threshold=None)
