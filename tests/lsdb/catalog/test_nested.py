import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype

import lsdb
import lsdb.nested as nd
from lsdb import Catalog


def test_nested_columns_property(small_sky_with_nested_sources):
    assert list(small_sky_with_nested_sources.nested_columns) == ["sources"]


def test_nest_lists(small_sky_with_nested_sources):
    """Test the behavior of catalog.nest_lists"""
    cat_ndf = small_sky_with_nested_sources._ddf.map_partitions(
        lambda df: df.set_index(df.index.to_numpy() + np.arange(len(df)))
    )
    catlists_ndf = cat_ndf.sources.nest.to_lists()
    smallsky_lists = cat_ndf[["id", "ra", "dec"]].join(catlists_ndf)
    small_sky_with_nested_sources._ddf = smallsky_lists
    cat_ndf_renested = small_sky_with_nested_sources.nest_lists(base_columns=["id", "ra", "dec"])

    # check column structure
    assert "nested" in cat_ndf_renested.columns
    assert "id" in cat_ndf_renested.columns
    assert "ra" in cat_ndf_renested.columns
    assert "dec" in cat_ndf_renested.columns
    assert cat_ndf_renested._ddf["nested"].nest.fields == cat_ndf["sources"].nest.fields

    # try a compute call
    renested_flat = cat_ndf_renested.compute()["nested"].nest.to_flat()
    original_flat = cat_ndf.compute()["sources"].nest.to_flat()

    pd.testing.assert_frame_equal(renested_flat, original_flat)


def test_nest_lists_only_list_columns(small_sky_with_nested_sources):
    """Test the behavior of catalog.nest_lists when only list columns are provided"""
    cat_ndf = small_sky_with_nested_sources._ddf.map_partitions(
        lambda df: df.set_index(df.index.to_numpy() + np.arange(len(df)))
    )
    catlists_ndf = cat_ndf.sources.nest.to_lists()
    smallsky_lists = cat_ndf[["id", "ra", "dec"]].join(catlists_ndf)
    small_sky_with_nested_sources._ddf = smallsky_lists

    # Use the columns from the original catalog as the list columns. All other
    # columns are inferred to be "base" columns.
    cat_ndf_renested = small_sky_with_nested_sources.nest_lists(list_columns=cat_ndf["sources"].nest.fields)

    # check column structure
    assert "nested" in cat_ndf_renested.columns
    assert "id" in cat_ndf_renested.columns
    assert "ra" in cat_ndf_renested.columns
    assert "dec" in cat_ndf_renested.columns
    assert cat_ndf_renested._ddf["nested"].nest.fields == cat_ndf["sources"].nest.fields

    # try a compute call
    renested_flat = cat_ndf_renested.compute()["nested"].nest.to_flat()
    original_flat = cat_ndf.compute()["sources"].nest.to_flat()

    pd.testing.assert_frame_equal(renested_flat, original_flat)


def test_reduce(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    assert reduced_cat.hc_structure.catalog_info.ra_column == ""
    assert reduced_cat.hc_structure.catalog_info.dec_column == ""

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    pd.testing.assert_frame_equal(reduced_cat_compute, reduced_ddf.compute())


def test_reduce_append_columns(small_sky_with_nested_sources):
    def mean_mag(mag):
        return {"mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={"mean_mag": float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(mean_mag, "sources.mag", meta={"mean_mag": float})

    pd.testing.assert_series_equal(reduced_cat_compute["mean_mag"], reduced_ddf.compute()["mean_mag"])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )


def test_reduce_no_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return np.mean(mag)

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={0: float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(mean_mag, "sources.mag", meta={0: float})

    pd.testing.assert_series_equal(reduced_cat_compute[0], reduced_ddf.compute()[0])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )


def test_reduce_invalid_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return pd.DataFrame.from_dict({"mean_mag": [np.mean(mag)]})

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={0: float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    with pytest.raises(ValueError):
        reduced_cat.compute()


def test_reduce_append_columns_raises_error(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    with pytest.raises(ValueError):
        small_sky_with_nested_sources.reduce(
            mean_mag,
            "ra",
            "dec",
            "sources.mag",
            meta={"ra": float, "dec": float, "mean_mag": float},
            append_columns=True,
        ).compute()


def test_reduce_infer_nesting(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        ra = np.asarray(ra)
        dec = np.asarray(dec)
        mag = np.asarray(mag)
        return {"new_nested.ra_mag": ra + mag, "new_nested.dec_mag": dec + mag}

    # With inferred nesting:
    # create a NestedDtype for the nested column "new_nested"
    new_dtype = NestedDtype(
        pa.struct([pa.field("ra_mag", pa.list_(pa.float64())), pa.field("dec_mag", pa.list_(pa.float64()))])
    )
    # use the lc_dtype in meta creation
    true_meta = npd.NestedFrame({"new_nested": pd.Series([], dtype=new_dtype)})

    res_true = small_sky_with_nested_sources.reduce(mean_mag, "ra", "dec", "sources.mag", meta=true_meta)

    assert "new_nested" in res_true.columns and "new_nested" in res_true._ddf.nested_columns
    assert list(res_true["new_nested"].nest.fields) == ["ra_mag", "dec_mag"]

    # Without inferred nesting:
    false_meta = (
        small_sky_with_nested_sources.compute()
        .reduce(mean_mag, "ra", "dec", "sources.mag", infer_nesting=False)
        .head(0)
    )

    res_false = small_sky_with_nested_sources.reduce(
        mean_mag, "ra", "dec", "sources.mag", infer_nesting=False, meta=false_meta
    )

    assert list(res_false.columns) == ["new_nested.ra_mag", "new_nested.dec_mag"]


def test_serialization_read(small_sky_with_nested_sources):
    assert isinstance(small_sky_with_nested_sources.dtypes["sources"], NestedDtype)


def test_serialization_round_trip(tmp_path, small_sky_order1_catalog, small_sky_source_catalog):
    cat = small_sky_order1_catalog.join_nested(
        small_sky_source_catalog, left_on="id", right_on="object_id", nested_column_name="lc"
    )
    assert isinstance(cat.dtypes["lc"], NestedDtype)
    out_path = tmp_path / "test_cat"
    cat.to_hats(out_path)
    read_catalog = lsdb.open_catalog(out_path)
    assert isinstance(read_catalog.dtypes["lc"], NestedDtype)
    assert isinstance(read_catalog.compute().dtypes["lc"], NestedDtype)


def test_getitem(small_sky_with_nested_sources):
    cat = small_sky_with_nested_sources[["ra", "dec", "sources.source_id", "sources.source_ra"]]
    assert np.all(cat.columns == ["ra", "dec", "sources"])
    assert cat.dtypes["sources"].field_names == ["source_id", "source_ra"]
    computed = cat.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])


def test_getitem_errors(small_sky_with_nested_sources):
    with pytest.raises(KeyError):
        _ = small_sky_with_nested_sources[["ra", "dec", "sources.source_id", "sources.wrong"]]
    with pytest.raises(KeyError):
        _ = small_sky_with_nested_sources[["ra", "wrong", "sources.source_id", "sources.source_ra"]]
    with pytest.raises(KeyError):
        _ = small_sky_with_nested_sources[["ra", "dec", "wrong.source_id", "sources.source_ra"]]


def test_nested_columns_after_filter(small_sky_with_nested_sources_dir):
    cat = lsdb.open_catalog(
        small_sky_with_nested_sources_dir, columns=["ra", "dec", "sources.source_id", "sources.source_ra"]
    )
    assert np.all(cat.columns == ["ra", "dec", "sources"])
    assert cat.dtypes["sources"].field_names == ["source_id", "source_ra"]
    computed = cat.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])
    filtered = cat.cone_search(300, -40, 10000)
    assert np.all(filtered.columns == ["ra", "dec", "sources"])
    assert filtered.dtypes["sources"].field_names == ["source_id", "source_ra"]
    computed = filtered.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])
