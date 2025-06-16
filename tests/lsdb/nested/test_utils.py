"""
Unit tests for utility functions in the lsdb.nested module.

This module tests the behavior of utility functions, such as the `count_nested`
wrapper, ensuring they behave as expected and are consistent with the
nested-pandas library.
"""

import pytest
from nested_pandas.utils import count_nested
from pandas.testing import assert_frame_equal

import lsdb.nested as nd


@pytest.mark.parametrize("join", [True, False])
@pytest.mark.parametrize("by", [None, "band"])
def test_count_nested(test_dataset, join, by):
    """test the count_nested wrapper"""

    # count_nested functionality is tested on the nested-pandas side
    # let's just make sure the behavior here is identical.

    result_dsk = nd.utils.count_nested(test_dataset, "nested", join=join, by=by).compute()
    result_pd = count_nested(test_dataset.compute(), "nested", join=join, by=by)

    assert_frame_equal(result_dsk, result_pd)
