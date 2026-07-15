import logging
import warnings

import nested_pandas as npd
import pandas as pd

from lsdb.operations.functions.merge_catalog_functions import (
    concat_partition_and_margin,
    create_merged_catalog_info,
)


def test_concat_partition_and_margin_empty_margin_no_warning():
    """An empty (but not None) margin must not raise the pandas empty/all-NA
    concatenation FutureWarning, and the partition is returned unchanged."""
    partition = npd.NestedFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    # An empty margin cache can carry columns with a different (object) dtype,
    # which is what triggers the FutureWarning under pd.concat.
    empty_margin = npd.NestedFrame({"a": pd.Series([], dtype="object"), "b": pd.Series([], dtype="object")})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = concat_partition_and_margin(partition, empty_margin)

    assert len(result) == len(partition)


def test_concat_partition_and_margin_nonempty_margin_concatenates():
    """A non-empty margin is still concatenated below the partition."""
    partition = npd.NestedFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    margin = npd.NestedFrame({"a": [7, 8], "b": [7.0, 8.0]})

    result = concat_partition_and_margin(partition, margin)

    assert len(result) == len(partition) + len(margin)


def test_create_merged_catalog_info_suffix_logging(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    with caplog.at_level(logging.WARNING):
        merged_catalog_info = create_merged_catalog_info(
            small_sky_catalog,
            small_sky_xmatch_catalog,
            "merged_catalog",
            ("_left", "_right"),
            suffix_method="overlapping_columns",
        )
        assert merged_catalog_info.catalog_name == "merged_catalog"
        assert merged_catalog_info.ra_column == "ra_left"
        assert merged_catalog_info.dec_column == "dec_left"
        assert "Renaming overlapping columns" not in caplog.text
