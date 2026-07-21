import pytest

from lsdb.io.common import round_sig


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        0.25,
        1 / 12,
        1e-10,
        0.99999999,
        8.765432e-300,
        123456789.123,
        0.000012345678,
        5.0,
        100000.0,
        99999.5,
        0.1,
        -0.1,
        -1 / 3,
    ],
)
def test_round_sig_matches_string_formatting(value):
    assert round_sig(value) == float(f"{value:.5g}")


@pytest.mark.parametrize("exponent", range(-20, 21))
def test_round_sig_matches_string_formatting_across_magnitudes(exponent):
    for sign in (1, -1):
        value = sign * (10**exponent) * 1.23456789
        assert round_sig(value) == float(f"{value:.5g}")


def test_round_sig_custom_digits():
    assert round_sig(1 / 3, digits=2) == float(f"{1 / 3:.2g}")
    assert round_sig(1 / 3, digits=8) == float(f"{1 / 3:.8g}")
