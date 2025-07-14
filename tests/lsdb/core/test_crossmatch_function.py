import nested_pandas as npd
import numpy as np
import pytest

import lsdb
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


@pytest.mark.parametrize("algo", [KdTreeCrossmatch, BuiltInCrossmatchAlgorithm.KD_TREE])
@pytest.mark.parametrize(
    "left, right",
    [
        ("dataframe", "dataframe"),
        ("dataframe", "catalog"),
        ("catalog", "dataframe"),
        ("catalog", "catalog"),
    ],
)
def test_dataframe_or_catalog_crossmatch(
    algo, left, right, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct
):
    # Determine which inputs need to be computed
    left_data = small_sky_catalog.compute() if left == "dataframe" else small_sky_catalog
    right_data = small_sky_xmatch_catalog.compute() if right == "dataframe" else small_sky_xmatch_catalog

    # Determine which args to pass
    left_args = {} if left == "catalog" else {"margin_threshold": 100}

    # Perform the crossmatch
    result = lsdb.crossmatch(
        left_data,
        right_data,
        suffixes=["_left", "_right"],
        algorithm=algo,
        radius_arcsec=0.01 * 3600,
        left_args=left_args,
        right_args={},
    ).compute()

    # Assertions
    assert isinstance(result, npd.NestedFrame)
    assert len(result) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in result["id_left"].to_numpy()
        xmatch_row = result[result["id_left"] == correct_row["ss_id"]]
        assert xmatch_row["id_right"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


@pytest.mark.parametrize("algo", [KdTreeCrossmatch])
@pytest.mark.parametrize(
    "left, right",
    [
        ("catalog", "invalid"),
        ("invalid", "catalog"),
    ],
)
def test_invalid_type_crossmatch(algo, left, right, small_sky_catalog, small_sky_xmatch_catalog):
    """Raise error if the type of left or right is invalid."""
    if left == "invalid":
        with pytest.raises(TypeError, match="Argument must be"):
            lsdb.crossmatch(np.array([1, 2, 3]), small_sky_xmatch_catalog, algorithm=algo)
        return
    if right == "invalid":
        with pytest.raises(TypeError, match="Argument must be"):
            lsdb.crossmatch(small_sky_catalog, np.array([1, 2, 3]), algorithm=algo)
        return


@pytest.mark.parametrize("algo", [KdTreeCrossmatch])
def test_invalid_margin_args_crossmatch(algo, small_sky_catalog, small_sky_xmatch_catalog):
    """Raise an error if an impossible margin argument combination is given."""
    with pytest.raises(
        ValueError, match="If require_right_margin is True, margin_threshold must not be None."
    ):
        lsdb.crossmatch(
            small_sky_catalog,
            small_sky_xmatch_catalog,
            algorithm=algo,
            require_right_margin=True,
            right_args={"margin_threshold": None},
        )


@pytest.mark.parametrize("algo", [KdTreeCrossmatch])
def test_ra_dec_columns_crossmatch(algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
    """Raise an error if an impossible margin argument combination is given."""

    # Compute dataframes
    left_dataframe = small_sky_catalog.compute()
    right_dataframe = small_sky_xmatch_catalog.compute()

    # Rename ra and dec columns to "RA" and "DEC"
    right_dataframe_caps_cols = right_dataframe.rename(columns={"ra": "RA"}).rename(columns={"dec": "DEC"})
    result = lsdb.crossmatch(
        left_dataframe,
        right_dataframe_caps_cols,
        algorithm=algo,
        radius_arcsec=0.01 * 3600,
        right_args={"margin_threshold": 100},
    ).compute()
    assert isinstance(result, npd.NestedFrame)
    assert len(result) == len(xmatch_correct)

    # Rename ra and dec columns to "Ra" and "Dec"
    right_dataframe_title_case_cols = right_dataframe.rename(columns={"ra": "Ra"}).rename(
        columns={"dec": "Dec"}
    )
    result = lsdb.crossmatch(
        left_dataframe,
        right_dataframe_title_case_cols,
        algorithm=algo,
        radius_arcsec=0.01 * 3600,
        right_args={"margin_threshold": 100},
    ).compute()
    assert isinstance(result, npd.NestedFrame)
    assert len(result) == len(xmatch_correct)

    # Rename ra and dec columns to abnormal names
    right_dataframe_abnormal_ra_col = right_dataframe.rename(columns={"ra": "abnormal_ra_col_name"})
    right_dataframe_abnormal_dec_col = right_dataframe.rename(columns={"dec": "abnormal_dec_col_name"})

    # Crossmatch method attempts to use default column names and fails
    with pytest.raises(ValueError, match="No column found for ra"):
        lsdb.crossmatch(
            left_dataframe,
            right_dataframe_abnormal_ra_col,
            algorithm=algo,
        )
    with pytest.raises(ValueError, match="No column found for dec"):
        lsdb.crossmatch(
            left_dataframe,
            right_dataframe_abnormal_dec_col,
            algorithm=algo,
        )

    # Successful crossmatch when we specify the abnormal RA column name in the right_args dict
    result = lsdb.crossmatch(
        left_dataframe,
        right_dataframe_abnormal_ra_col,
        algorithm=algo,
        radius_arcsec=0.01 * 3600,
        right_args={
            "margin_threshold": 100,
            "ra_column": "abnormal_ra_col_name",
        },
    ).compute()
    assert isinstance(result, npd.NestedFrame)
    assert len(result) == len(xmatch_correct)

    # And finally, check we can override both left and right RA, dec column names using the ra/dec_column args
    right_dataframe_abnormal_cols = right_dataframe.rename(
        columns={"ra": "abnormal_ra_col_name", "dec": "abnormal_dec_col_name"}
    )
    left_dataframe_abnormal_cols = left_dataframe.rename(
        columns={"ra": "abnormal_ra_col_name", "dec": "abnormal_dec_col_name"}
    )
    result = lsdb.crossmatch(
        left_dataframe_abnormal_cols,
        right_dataframe_abnormal_cols,
        algorithm=algo,
        radius_arcsec=0.01 * 3600,
        ra_column="abnormal_ra_col_name",
        dec_column="abnormal_dec_col_name",
        right_args={
            "margin_threshold": 100,
            "ra_column": "abnormal_ra_col_name",
        },
    ).compute()
    assert isinstance(result, npd.NestedFrame)
    assert len(result) == len(xmatch_correct)
