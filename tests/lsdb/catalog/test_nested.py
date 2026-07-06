import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype

import lsdb
from lsdb import Catalog


def test_nested_columns_property(small_sky_with_nested_sources):
    assert list(small_sky_with_nested_sources.nested_columns) == ["sources"]


def test_nest_lists(small_sky_with_nested_sources):
    """Test the behavior of catalog.nest_lists"""

    def listify_sources(nf):
        nf_result = nf.join(nf["sources"].to_lists())
        # Need to set dtypes properly
        return nf_result.astype(
            {
                "source_id": pd.ArrowDtype(pa.list_(pa.int64())),
                "source_ra": pd.ArrowDtype(pa.list_(pa.float64())),
                "source_dec": pd.ArrowDtype(pa.list_(pa.float64())),
                "mag": pd.ArrowDtype(pa.list_(pa.float64())),
            }
        )

    meta_sampler = small_sky_with_nested_sources.head(1)
    meta = listify_sources(meta_sampler)
    list_sources = small_sky_with_nested_sources.map_partitions(listify_sources, meta=meta)

    cat_renested = list_sources.nest_lists(
        name="repacked", list_columns=["source_id", "source_ra", "source_dec", "mag"]
    )

    assert "repacked.source_id" in cat_renested.exploded_columns
    assert "repacked.source_ra" in cat_renested.exploded_columns
    assert "repacked.source_dec" in cat_renested.exploded_columns
    assert "repacked.mag" in cat_renested.exploded_columns
    assert "repacked.band" not in cat_renested.exploded_columns

    # try a compute call
    cat_renested_ndf = cat_renested.compute()
    assert isinstance(cat_renested_ndf, npd.NestedFrame)
    assert len(cat_renested_ndf["sources"]) == len(cat_renested_ndf["repacked"])


def test_map_rows(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.map_rows(
        mean_mag,
        columns=["ra", "dec", "sources.mag"],
        row_container="args",
        meta={"ra": float, "dec": float, "mean_mag": float},
    )
    assert isinstance(reduced_cat, Catalog)

    assert reduced_cat.hc_structure.catalog_info.ra_column == "ra"
    assert reduced_cat.hc_structure.catalog_info.dec_column == "dec"

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    def map_row_part(nf):
        return nf.map_rows(
            mean_mag,
            columns=["ra", "dec", "sources.mag"],
            row_container="args",
        )

    reduced_mp_cat = small_sky_with_nested_sources.map_partitions(
        map_row_part, meta=reduced_cat_compute.head(0)
    )

    pd.testing.assert_frame_equal(reduced_cat_compute, reduced_mp_cat.compute())


def test_map_rows_meta_required(small_sky_with_nested_sources):
    with pytest.raises(ValueError, match="specify `meta`"):
        small_sky_with_nested_sources.map_rows(
            lambda mag: {"mean_mag": np.mean(mag)}, columns=["sources.mag"], meta=None
        )


def test_map_rows_append_columns(small_sky_with_nested_sources):
    def mean_mag(mag):
        return {"mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.map_rows(
        mean_mag, columns=["sources.mag"], row_container="args", meta={"mean_mag": float}, append_columns=True
    )
    assert isinstance(reduced_cat, Catalog)
    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    def map_row_part(nf):
        return nf.map_rows(mean_mag, columns=["sources.mag"], row_container="args", append_columns=True)

    reduced_mp_cat = small_sky_with_nested_sources.map_partitions(
        map_row_part, meta=reduced_cat_compute.head(0)
    )
    pd.testing.assert_series_equal(reduced_cat_compute["mean_mag"], reduced_mp_cat.compute()["mean_mag"])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )

    def create_subcolumn(ra, source_ra):
        """UDF that returns a dictionary"""
        return {"sources.t_ra": source_ra - ra}

    reduced_cat = small_sky_with_nested_sources.map_rows(
        create_subcolumn,
        columns=["ra", "sources.source_ra"],
        row_container="args",
        meta={"sources.t_ra": float},
        append_columns=True,
    )
    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    def map_row_part_2(nf):
        return nf.map_rows(
            create_subcolumn,
            columns=["ra", "sources.source_ra"],
            row_container="args",
            append_columns=True,
        )

    reduced_mp_cat = small_sky_with_nested_sources.map_partitions(
        map_row_part_2, meta=reduced_cat_compute.head(0)
    )
    expected_t_ra = reduced_mp_cat.compute()["sources.t_ra"]
    pd.testing.assert_series_equal(expected_t_ra, reduced_cat_compute["sources.t_ra"])

    def create_subcolumn_2(ra, source_ra):
        """Same UDF, but now returning an array"""
        return source_ra - ra

    reduced_cat = small_sky_with_nested_sources.map_rows(
        create_subcolumn_2,
        columns=["ra", "sources.source_ra"],
        output_names=["sources.t_ra"],
        row_container="args",
        meta={"sources.t_ra": float},
        append_columns=True,
    )
    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    pd.testing.assert_series_equal(expected_t_ra, reduced_cat_compute["sources.t_ra"])


def test_map_rows_append_columns_raises_error_on_overlap(small_sky_with_nested_sources):
    def calc_magerr(obj_id, ra_err, dec_err):
        return {"id": obj_id, "mean_err": abs(ra_err - dec_err) / 2}

    with pytest.raises(ValueError, match="already exist"):
        small_sky_with_nested_sources.map_rows(
            calc_magerr,
            columns=["id", "ra_error", "dec_error"],
            row_container="args",
            meta={"id": float, "mean_err": float},  # "id" column already exists in the catalog
            append_columns=True,
        )


def test_map_rows_append_columns_raises_error_with_full_meta(small_sky_with_nested_sources):
    def calc_magerr(ra_err, dec_err):
        return {"mean_err": abs(ra_err - dec_err) / 2}

    meta = small_sky_with_nested_sources.meta.copy()
    meta["mean_err"] = np.float64(0)

    with pytest.raises(ValueError, match="already exist"):
        small_sky_with_nested_sources.map_rows(
            calc_magerr,
            columns=["ra_error", "dec_error"],
            row_container="args",
            meta=meta,  # should not be full meta when `append_columns` is set
            append_columns=True,
        )


def test_map_rows_no_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return np.mean(mag)

    reduced_cat = small_sky_with_nested_sources.map_rows(
        mean_mag,
        columns=["sources.mag"],
        row_container="args",
        meta={0: float},
        append_columns=True,
    )

    assert isinstance(reduced_cat, Catalog)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    def map_row_part(nf):
        return nf.map_rows(
            mean_mag,
            columns=["sources.mag"],
            row_container="args",
            append_columns=True,
        )

    reduced_mp_cat = small_sky_with_nested_sources.map_partitions(
        map_row_part, meta=reduced_cat_compute.head(0)
    )

    pd.testing.assert_series_equal(reduced_cat_compute[0], reduced_mp_cat.compute()[0])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )


def test_map_rows_invalid_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return pd.DataFrame.from_dict({"mean_mag": [np.mean(mag)]})

    reduced_cat = small_sky_with_nested_sources.map_rows(
        mean_mag,
        columns=["sources.mag"],
        row_container="args",
        meta={0: float},
        append_columns=True,
    )
    assert isinstance(reduced_cat, Catalog)

    with pytest.raises(RuntimeError):
        reduced_cat.compute()


def test_map_rows_infer_nesting(small_sky_with_nested_sources):
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

    res_true = small_sky_with_nested_sources.map_rows(
        mean_mag,
        columns=["ra", "dec", "sources.mag"],
        row_container="args",
        meta=true_meta,
    )

    assert "new_nested" in res_true.columns
    assert "new_nested.ra_mag" in res_true.exploded_columns
    assert "new_nested.dec_mag" in res_true.exploded_columns

    # Without inferred nesting:
    false_meta = (
        small_sky_with_nested_sources.compute()
        .map_rows(
            mean_mag,
            columns=["ra", "dec", "sources.mag"],
            row_container="args",
            infer_nesting=False,
        )
        .head(0)
    )

    res_false = small_sky_with_nested_sources.map_rows(
        mean_mag,
        columns=["ra", "dec", "sources.mag"],
        row_container="args",
        infer_nesting=False,
        meta=false_meta,
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
    cat.write_catalog(out_path)
    read_catalog = lsdb.open_catalog(out_path)
    assert isinstance(read_catalog.dtypes["lc"], NestedDtype)
    assert isinstance(read_catalog.compute().dtypes["lc"], NestedDtype)


def test_getitem(small_sky_with_nested_sources):
    cat = small_sky_with_nested_sources[["ra", "dec", "sources.source_id", "sources.source_ra"]]
    assert np.all(cat.columns == ["ra", "dec", "sources"])
    assert "sources.source_id" in cat.exploded_columns
    assert "sources.source_ra" in cat.exploded_columns
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
    assert "sources.source_id" in cat.exploded_columns
    assert "sources.source_ra" in cat.exploded_columns
    computed = cat.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])
    filtered = cat.cone_search(300, -40, 10000)
    assert np.all(filtered.columns == ["ra", "dec", "sources"])
    assert "sources.source_id" in filtered.exploded_columns
    assert "sources.source_ra" in filtered.exploded_columns
    computed = filtered.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])
