import nested_pandas as npd
import pandas as pd
from nested_pandas import utils as npd_utils

from ..core import NestedFrame


def count_nested(df, nested, by=None, join=True) -> NestedFrame:
    """Counts the number of rows of a nested dataframe.

    Wraps Nested-Pandas count_nested.

    Parameters
    ----------
    df: NestedFrame
        A NestedFrame that contains the desired `nested` series
        to count.
    nested: 'str'
        The label of the nested series to count.
    by: 'str', optional
        Specifies a column within nested to count by, returning
        a count for each unique value in `by`.
    join: bool, optional
        Join the output count columns to df and return df, otherwise
        just return a NestedFrame containing only the count columns.

    Returns
    -------
    NestedFrame
    """

    # The meta varies depending on the parameters

    # first depending on by
    if by is not None:
        # will have one column per unique value of the specified column
        # requires some computation to determine these values
        by_cols = sorted(df[nested].nest.to_flat()[by].unique())
        out_cols = [f"n_{nested}_{col}" for col in by_cols]
    else:
        # otherwise just have a single column output
        out_cols = [f"n_{nested}"]

    # add dtypes
    meta = npd.NestedFrame({col: 0 for col in out_cols}, index=[])

    # and second depending on join
    if join:
        # adds the meta onto the existing meta
        meta = pd.concat([df._meta, meta])

    return df.map_partitions(lambda x: npd_utils.count_nested(x, nested, by=by, join=join), meta=meta)
