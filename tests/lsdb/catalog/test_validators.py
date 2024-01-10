import numpy as np
import numpy.testing as npt
import pytest

from lsdb.core.search.polygon_search import get_cartesian_polygon
from lsdb.core.search.validators import CoordinatesValidator


def test_wrap_right_ascension():
    # Does not need to be wrapped
    ra = [20, 150, 350]
    wrapped_ra = CoordinatesValidator._wrap_ra_values(ra)
    assert np.array_equal(np.array(ra), wrapped_ra)
    # Wrap a single value
    assert 1200 % 360 == CoordinatesValidator._wrap_ra_values(1200)
    # Some values are out of range, need to be wrapped
    ra, expected_ra = [-20, 150, 480], [340, 150, 120]
    wrapped_ra = CoordinatesValidator._wrap_ra_values(ra)
    assert np.array_equal(np.array(expected_ra), wrapped_ra)
    # If the values fall on the discontinuity
    ra, expected_ra = [0, 360], [0, 0]
    wrapped_ra = CoordinatesValidator._wrap_ra_values(ra)
    assert np.array_equal(np.array(expected_ra), wrapped_ra)


def test_validate_declination():
    error_msg = "dec must be between -90 and 90"
    with pytest.raises(ValueError, match=error_msg):
        CoordinatesValidator._validate_dec_values(-100)
    with pytest.raises(ValueError, match=error_msg):
        CoordinatesValidator._validate_dec_values(100)
    with pytest.raises(ValueError, match=error_msg):
        CoordinatesValidator._validate_dec_values([-80, 100])
    CoordinatesValidator._validate_dec_values([-80, 10])


def test_wrapped_polygon_coordinates():
    """Tests the scenario where the polygon edges intersect the
    discontinuity of the RA [0,360] degrees range. For the same
    polygon we can have four possible combination of coordinates:
        - [(-20, 1), (-20, -1), (20, -1), (20, 1)]
        - [(-20, 1), (-20, -1), (380, -1), (380, 1)]
        - [(340, 1), (340, -1), (20, -1), (20, 1)]
        - [(340, 1), (340, -1), (-340, -1), (-340, 1)]
    """
    def assert_wrapped_polygon_is_the_same(vertices):
        ra, dec = np.array(vertices).T
        _, vertices_xyz = get_cartesian_polygon(vertices)
        transformed_ra = CoordinatesValidator._wrap_ra_values(ra)
        new_coords = list(zip(transformed_ra, dec))
        _, vertices_xyz_2 = get_cartesian_polygon(new_coords)
        npt.assert_allclose(vertices_xyz, vertices_xyz_2, rtol=1e-7)
    vertices_1 = [(-20, 1), (-20, -1), (20, -1), (20, 1)]
    assert_wrapped_polygon_is_the_same(vertices_1)
    vertices_2 = [(-20, 1), (-20, -1), (380, -1), (380, 1)]
    assert_wrapped_polygon_is_the_same(vertices_2)
    vertices_3 = [(340, 1), (340, -1), (20, -1), (20, 1)]
    assert_wrapped_polygon_is_the_same(vertices_3)
    vertices_4 = [(340, 1), (340, -1), (-340, -1), (-340, 1)]
    assert_wrapped_polygon_is_the_same(vertices_4)
