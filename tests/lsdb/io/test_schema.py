import pandas as pd
import pyarrow as pa
import pytest
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

from lsdb.io.schema import get_arrow_schema


def _dataframe(columns):
    """Build a tiny DataFrame that carries the spatial index in its index."""
    values = {name: [1.0] for name in columns}
    index = pd.Index([9999], name=SPATIAL_INDEX_COLUMN)
    return pd.DataFrame(values, index=index)


def test_spatial_index_is_the_first_field():
    """Issue #595: the spatial index column is always the first schema field."""
    schema = get_arrow_schema(_dataframe(["ra", "dec"]))
    assert schema.names == [SPATIAL_INDEX_COLUMN, "ra", "dec"]


def test_remaining_fields_keep_their_relative_order():
    """Moving the spatial index field to the front leaves the others as-is."""
    schema = get_arrow_schema(_dataframe(["ra", "dec", "mag", "id"]))
    assert schema.names == [SPATIAL_INDEX_COLUMN, "ra", "dec", "mag", "id"]


def test_field_order_is_unchanged_without_spatial_index():
    """DataFrames without the spatial index column are passed through untouched."""
    df = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
    assert get_arrow_schema(df).names == ["ra", "dec"]


def test_spatial_index_field_keeps_its_type():
    """Reordering the fields does not alter the type of the moved field."""
    df = _dataframe(["ra", "dec"])
    schema = get_arrow_schema(df)
    expected_type = pa.Schema.from_pandas(df).remove_metadata().field(SPATIAL_INDEX_COLUMN).type
    assert schema.field(SPATIAL_INDEX_COLUMN).type == expected_type == pa.int64()


def test_schema_metadata_stays_removed():
    """The reordering does not bring the pandas metadata back."""
    schema = get_arrow_schema(_dataframe(["ra", "dec"]))
    assert schema.metadata is None
    assert all(field.metadata is None for field in schema)


def test_spatial_index_as_leading_column_stays_leading():
    """A DataFrame that already leads with the spatial index is left alone."""
    df = pd.DataFrame({SPATIAL_INDEX_COLUMN: [9999], "ra": [1.0], "dec": [2.0]})
    schema = get_arrow_schema(df)
    assert schema.names == [SPATIAL_INDEX_COLUMN, "ra", "dec"]


@pytest.mark.parametrize(
    "index_name",
    [None, "id"],
    ids=["unnamed-index", "non-spatial-index"],
)
def test_non_spatial_index_does_not_move(index_name):
    """Other index names are not treated as the spatial index column."""
    df = pd.DataFrame({"ra": [1.0], "dec": [2.0]}, index=pd.Index([7], name=index_name))
    schema = get_arrow_schema(df)
    assert SPATIAL_INDEX_COLUMN not in schema.names
    # pyarrow keeps such an index as a trailing field, under its own name.
    assert schema.names[-1] == ("__index_level_0__" if index_name is None else "id")
