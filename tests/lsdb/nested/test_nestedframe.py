"""
Unit tests for the NestedFrame functionality in the lsdb.nested module.

This module tests the construction, manipulation, and I/O operations of
NestedFrame objects, including nested column handling and integration with
Dask and Pandas.
"""

import dask
import dask.dataframe as dd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas.series.dtype import NestedDtype
from pandas.testing import assert_frame_equal

import lsdb.nested as nd
from lsdb.nested.datasets import generate_data

dask.config.set({"dataframe.convert-string": False})


def test_nestedframe_construction(test_dataset):
    """test the construction of a nestedframe"""
    assert len(test_dataset) == 50
    assert test_dataset.columns.to_list() == ["a", "b", "nested"]
    assert isinstance(test_dataset["nested"].dtype, NestedDtype)


def test_nestedframe_from_dask_keeps_index_name():
    """test index name is set in from_dask_dataframe"""
    index_name = "test"
    a = pd.DataFrame({"a": [1, 2, 3]})
    a.index.name = index_name
    ddf = dd.from_pandas(a)
    assert ddf.index.name == index_name
    ndf = nd.NestedFrame.from_dask_dataframe(ddf)
    assert isinstance(ndf, nd.NestedFrame)
    assert ndf.index.name == index_name


def test_all_columns(test_dataset):
    """all_columns property test"""
    all_cols = test_dataset.all_columns

    assert all_cols["base"].to_list() == test_dataset.columns.to_list()
    assert all_cols["nested"] == ["t", "flux", "band"]


def test_nested_columns(test_dataset):
    """nested_columns property test"""
    assert test_dataset.nested_columns == ["nested"]


def test_getitem_on_nested():
    """test getitem with nested columns"""
    ndf = generate_data(10, 10, npartitions=3, seed=1)

    nest_col = ndf["nested.t"]

    assert len(nest_col) == 100
    assert nest_col.name == "t"


def test_set_or_replace_nested_col():
    """Test that __setitem__ can set or replace a column in a existing nested structure"""

    ndf = generate_data(10, 10, npartitions=3, seed=1)

    # test direct replacement, with ints
    orig_t_head = ndf["nested.t"].head(10, npartitions=-1)

    ndf["nested.t"] = ndf["nested.t"] + 1
    assert np.array_equal(ndf["nested.t"].head(10).values.to_numpy(), orig_t_head.values.to_numpy() + 1)

    # test direct replacement, with str
    ndf["nested.band"] = "lsst"
    assert np.all(ndf["nested.band"].compute().values.to_numpy() == "lsst")

    # test setting a new column within nested
    ndf["nested.t_plus_flux"] = ndf["nested.t"] + ndf["nested.flux"]

    true_vals = (ndf["nested.t"] + ndf["nested.flux"]).head(10).values.to_numpy()
    assert np.array_equal(ndf["nested.t_plus_flux"].head(10).values.to_numpy(), true_vals)


def test_set_new_nested_col():
    """Test that __setitem__ can create a new nested structure"""

    ndf = generate_data(10, 10, npartitions=3, seed=1)

    # assign column in new nested structure from columns in nested
    ndf["new_nested.t_plus_flux"] = ndf["nested.t"] + ndf["nested.flux"]

    assert "new_nested" in ndf.nested_columns
    assert "t_plus_flux" in ndf["new_nested"].nest.fields

    assert np.array_equal(
        ndf["new_nested.t_plus_flux"].compute().values.to_numpy(),
        ndf["nested.t"].compute().values.to_numpy() + ndf["nested.flux"].compute().values.to_numpy(),
    )


def test_add_nested(test_dataset_no_add_nested):
    """test the add_nested function"""
    base, layer = test_dataset_no_add_nested

    base_with_nested = base.add_nested(layer, "nested")

    # Check that the result is a nestedframe
    assert isinstance(base_with_nested, nd.NestedFrame)

    # Check that there's a new nested column with the correct dtype
    assert "nested" in base_with_nested.columns
    assert isinstance(base_with_nested.dtypes["nested"], NestedDtype)

    # Check that the nested partitions were used
    assert base_with_nested.npartitions == 10

    assert len(base_with_nested.compute()) == 50


def test_from_flat():
    """Test the from_flat wrapping, make sure meta is assigned correctly"""

    nf = nd.NestedFrame.from_pandas(
        npd.NestedFrame(
            {
                "a": [1, 1, 1, 2, 2, 2],
                "b": [2, 2, 2, 4, 4, 4],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [2, 4, 6, 8, 10, 12],
            },
            index=[0, 0, 0, 1, 1, 1],
        )
    )

    # Check full inputs
    ndf = nd.NestedFrame.from_flat(nf, base_columns=["a", "b"], nested_columns=["c", "d"])
    assert list(ndf.columns) == ["a", "b", "nested"]
    assert list(ndf["nested"].nest.fields) == ["c", "d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2

    # Check omitting a base column
    ndf = nd.NestedFrame.from_flat(nf, base_columns=["a"], nested_columns=["c", "d"])
    assert list(ndf.columns) == ["a", "nested"]
    assert list(ndf["nested"].nest.fields) == ["c", "d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2

    # Check omitting a nested column
    ndf = nd.NestedFrame.from_flat(nf, base_columns=["a", "b"], nested_columns=["d"])
    assert list(ndf.columns) == ["a", "b", "nested"]
    assert list(ndf["nested"].nest.fields) == ["d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2

    # Check no base columns
    ndf = nd.NestedFrame.from_flat(nf, base_columns=[], nested_columns=["c", "d"])
    assert list(ndf.columns) == ["nested"]
    assert list(ndf["nested"].nest.fields) == ["c", "d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2

    # Check inferred nested columns
    ndf = nd.NestedFrame.from_flat(nf, base_columns=["a", "b"])
    assert list(ndf.columns) == ["a", "b", "nested"]
    assert list(ndf["nested"].nest.fields) == ["c", "d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2

    # Check using an index
    ndf = nd.NestedFrame.from_flat(nf, base_columns=["b"], on="a")
    assert list(ndf.columns) == ["b", "nested"]
    assert list(ndf["nested"].nest.fields) == ["c", "d"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert len(ndf_comp) == 2


def test_from_lists():
    """Test the from_lists wrapping, make sure meta is assigned correctly"""

    nf = nd.NestedFrame.from_pandas(
        npd.NestedFrame(
            {
                "c": [1, 2, 3],
                "d": [2, 4, 6],
                "e": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "f": [["dog", "cat", "bird"], ["dog", "cat", "bird"], ["dog", "cat", "bird"]],
            },
            index=[0, 1, 2],
        )
    )
    nf = nf.astype({"e": pd.ArrowDtype(pa.list_(pa.int64())), "f": pd.ArrowDtype(pa.list_(pa.string()))})

    # Check with just base_columns
    ndf = nd.NestedFrame.from_lists(nf, base_columns=["c", "d"])
    assert list(ndf.columns) == ["c", "d", "nested"]
    assert list(ndf["nested"].nest.fields) == ["e", "f"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)
    assert_frame_equal(ndf_comp.iloc[:0], ndf.meta)

    # Check with just list_columns
    ndf = nd.NestedFrame.from_lists(nf, list_columns=["e", "f"])
    assert list(ndf.columns) == ["c", "d", "nested"]
    assert list(ndf["nested"].nest.fields) == ["e", "f"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)

    # Check with base subset
    ndf = nd.NestedFrame.from_lists(nf, base_columns=["c"], list_columns=["e", "f"])
    assert list(ndf.columns) == ["c", "nested"]
    assert list(ndf["nested"].nest.fields) == ["e", "f"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)

    # Check with list subset
    ndf = nd.NestedFrame.from_lists(nf, base_columns=["c", "d"], list_columns=["f"])
    assert list(ndf.columns) == ["c", "d", "nested"]
    assert list(ndf["nested"].nest.fields) == ["f"]
    ndf_comp = ndf.compute()
    assert list(ndf.columns) == list(ndf_comp.columns)
    assert list(ndf["nested"].nest.fields) == list(ndf["nested"].nest.fields)


def test_from_lists_errors():
    """test that the dtype errors are appropriately raised"""
    nf = nd.NestedFrame.from_pandas(
        npd.NestedFrame(
            {
                "c": [1, 2, 3],
                "d": [2, 4, 6],
                "e": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "f": [["dog", "cat", "bird"], ["dog", "cat", "bird"], ["dog", "cat", "bird"]],
            },
            index=[0, 1, 2],
        )
    )
    # first check for no list_column error
    with pytest.raises(ValueError):
        nd.NestedFrame.from_lists(nf, base_columns=["c", "d", "e", "f"])

    # next check for non-pyarrow dtype in list_column
    with pytest.raises(TypeError):
        nd.NestedFrame.from_lists(nf, base_columns=["e"])

    # And check for non-list pyarrow type in list_column
    nf = nf.astype({"d": pd.ArrowDtype(pa.int64())})
    with pytest.raises(TypeError):
        nd.NestedFrame.from_lists(nf, base_columns=["d"])


def test_query_on_base(test_dataset):
    """test the query function on base columns"""

    # Try a few basic queries
    assert len(test_dataset.query("a  > 0.5").compute()) == 22
    assert len(test_dataset.query("a  > 0.5 & b > 1").compute()) == 13
    assert len(test_dataset.query("a  > 2").compute()) == 0


def test_query_on_nested(test_dataset):
    """test the query function on nested columns"""

    # Try a few nested queries
    res = test_dataset.query("nested.flux>75").compute()
    assert len(res.loc[1]["nested"]) == 127

    res = test_dataset.query("nested.band == 'g'").compute()

    assert len(res.loc[1]["nested"]) == 232
    assert len(res) == 50  # make sure the base df remains unchanged


def test_sort_values(test_dataset):
    """test the sort_values function"""

    # test sorting on base columns
    sorted_base = test_dataset.sort_values(by="a")
    assert sorted_base["a"].values.compute().tolist() == sorted(test_dataset["a"].values.compute().tolist())

    # test sorting on nested columns
    sorted_nested = test_dataset.sort_values(by="nested.flux", ascending=False)
    assert sorted_nested.compute().iloc[0]["nested"]["flux"].values.tolist() == sorted(
        test_dataset.compute().iloc[0]["nested"]["flux"].values.tolist(),
        reverse=True,
    )
    assert sorted_nested.known_divisions  # Divisions should be known

    # Make sure we trigger multi-target exception
    with pytest.raises(ValueError):
        test_dataset.sort_values(by=["a", "nested.flux"])


def test_reduce(test_dataset):
    """test the reduce function"""

    def reflect_inputs(*args):
        return args

    res = test_dataset.reduce(reflect_inputs, "a", "nested.t", meta={0: float, 1: float})

    assert len(res) == 50
    assert isinstance(res.compute().loc[0][0], float)
    assert isinstance(res.compute().loc[0][1], np.ndarray)

    res2 = test_dataset.reduce(np.mean, "nested.flux", meta={0: float})

    assert pytest.approx(res2.compute()[0][15], 0.1) == 53.635174
    assert pytest.approx(sum(res2.compute()[0]), 0.1) == 2488.960119


@pytest.mark.parametrize("meta", ["df", "series"])
def test_reduce_output_type(meta):
    """test the meta handling of reduce"""

    a = npd.NestedFrame({"a": pd.Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))}, index=[0, 0, 1])
    b = npd.NestedFrame({"b": pd.Series([1, 2], dtype=pd.ArrowDtype(pa.int64()))}, index=[0, 1])

    ndf = b.add_nested(a, name="test")
    nddf = nd.NestedFrame.from_pandas(ndf, npartitions=1)

    if meta == "df":

        def mean_arr(b, arr):  # type: ignore
            return {"b": b, "mean": np.mean(arr)}  # type: ignore

        reduced = nddf.reduce(mean_arr, "b", "test.a", meta={"b": int, "mean": float})
    elif meta == "series":

        def mean_arr(arr):  # type: ignore
            return np.mean(arr)  # type: ignore

        reduced = nddf.reduce(mean_arr, "test.a", meta=(0, "float"))
    else:
        reduced = None
    assert isinstance(reduced, nd.NestedFrame)
    assert isinstance(reduced.compute(), npd.NestedFrame)


def test_reduce_output_inference():
    """test the extension of the reduce result nesting inference"""

    ndd = generate_data(20, 20, npartitions=2, seed=1)

    def complex_output(flux):
        return {
            "max_flux": np.max(flux),
            "lc.flux_quantiles": np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "lc.labels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "meta.colors": ["green", "red", "blue"],
        }

    # this sucks
    result_meta = npd.NestedFrame(
        {
            "max_flux": pd.Series([], dtype="float"),
            "lc": pd.Series(
                [],
                dtype=NestedDtype(
                    pa.struct(
                        [
                            pa.field("flux_quantiles", pa.list_(pa.float64())),
                            pa.field("labels", pa.list_(pa.float64())),
                        ]
                    )
                ),
            ),
            "meta": pd.Series([], dtype=NestedDtype(pa.struct([pa.field("colors", pa.list_(pa.string()))]))),
        }
    )
    result = ndd.reduce(complex_output, "nested.flux", infer_nesting=True, meta=result_meta)

    assert list(result.dtypes) == list(result.compute().dtypes)
    assert list(result.columns) == list(result.compute().columns)


def test_to_parquet_combined(test_dataset, tmp_path):
    """test to_parquet when saving all layers to a single directory"""

    test_save_path = tmp_path / "test_dataset"

    # send to parquet
    test_dataset.to_parquet(test_save_path, by_layer=False)

    # load back from parquet
    loaded_dataset = nd.read_parquet(test_save_path, calculate_divisions=True)
    # we should file bug for this and investigate
    loaded_dataset = loaded_dataset.reset_index().set_index("index")

    # Check for equivalence
    assert test_dataset.divisions == loaded_dataset.divisions

    test_dataset = test_dataset.compute()
    loaded_dataset = loaded_dataset.compute()

    assert test_dataset.equals(loaded_dataset)


def test_to_parquet_by_layer(test_dataset, tmp_path):
    """test to_parquet when saving layers to subdirectories"""

    test_save_path = tmp_path / "test_dataset"

    # send to parquet
    test_dataset.to_parquet(test_save_path, by_layer=True, write_index=True)

    # load back from parquet
    loaded_base = nd.read_parquet(test_save_path / "base", calculate_divisions=True)
    loaded_nested = nd.read_parquet(test_save_path / "nested", calculate_divisions=True)

    # this is read as a large_string, just make it a string
    loaded_nested = loaded_nested.astype({"band": pd.ArrowDtype(pa.string())})
    loaded_dataset = loaded_base.add_nested(loaded_nested, "nested")

    # Check for equivalence
    assert test_dataset.divisions == loaded_dataset.divisions

    test_dataset = test_dataset.compute()
    loaded_dataset = loaded_dataset.compute()

    assert test_dataset.equals(loaded_dataset)


def test_from_epyc():
    """test a dataset from epyc. Motivated by https://github.com/lincc-frameworks/nested-dask/issues/21"""
    # Load some ZTF data
    catalogs_dir = "https://epyc.astro.washington.edu/~lincc-frameworks/half_degree_surveys/ztf/"

    object_ndf = (
        nd.read_parquet(f"{catalogs_dir}/ztf_object", columns=["ra", "dec", "ps1_objid"])
        .set_index("ps1_objid", sort=True)
        .persist()
    )

    source_ndf = (
        nd.read_parquet(
            f"{catalogs_dir}/ztf_source", columns=["mjd", "mag", "magerr", "band", "ps1_objid", "catflags"]
        )
        .set_index("ps1_objid", sort=True)
        .persist()
    )

    object_ndf = object_ndf.add_nested(source_ndf, "ztf_source")

    # Apply a mean function
    meta = pd.DataFrame(columns=[0], dtype=float)
    result = object_ndf.reduce(np.mean, "ztf_source.mag", meta=meta).compute()

    # just make sure the result was successfully computed
    assert len(result) == 9817


@pytest.mark.parametrize("pkg", ["pandas", "nested-pandas"])
@pytest.mark.parametrize("with_nested", [True, False])
def test_from_pandas(pkg, with_nested):
    """Test that from_pandas returns a NestedFrame"""

    if pkg == "pandas":
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
    elif pkg == "nested-pandas":
        df = npd.NestedFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
        if with_nested:
            nested = npd.NestedFrame({"b": [5, 10, 15, 20, 25, 30]}, index=[1, 1, 2, 2, 3, 3])
            df = df.add_nested(nested, "nested")

    ndf = nd.NestedFrame.from_pandas(df)
    assert isinstance(ndf, nd.NestedFrame)


@pytest.mark.parametrize("with_nested", [True, False])
def test_from_delayed(with_nested):
    """Test that from_delayed returns a NestedFrame"""

    nf = nd.datasets.generate_data(10, 10)
    if not with_nested:
        nf = nf.drop("nested", axis=1)

    delayed = nf.to_delayed()

    ndf = nd.NestedFrame.from_delayed(dfs=delayed, meta=nf._meta)  # pylint: disable=protected-access
    assert isinstance(ndf, nd.NestedFrame)


def test_from_map(test_dataset, tmp_path):
    """Test that from_map returns a NestedFrame"""

    # Setup a temporary directory for files
    test_save_path = tmp_path / "test_dataset"

    # Save Base to Parquet
    test_dataset[["a", "b"]].to_parquet(test_save_path, write_index=True)

    # Load from_map
    paths = [
        tmp_path / "test_dataset" / "0.parquet",
        tmp_path / "test_dataset" / "1.parquet",
        tmp_path / "test_dataset" / "2.parquet",
    ]

    result_meta = test_dataset[["a", "b"]]._meta  # pylint: disable=protected-access
    ndf = nd.NestedFrame.from_map(nd.read_parquet, paths, meta=result_meta)
    assert isinstance(ndf, nd.NestedFrame)
