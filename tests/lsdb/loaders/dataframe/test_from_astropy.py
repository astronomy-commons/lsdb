import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

import lsdb


def test_from_astropy_table():
    data = {
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "magnitude": [15.0, 16.5, 14.2],
    }
    table = Table(data)
    catalog = lsdb.from_astropy(table, ra_column="ra", dec_column="dec")

    assert catalog is not None
    assert len(catalog) == 3
    assert "magnitude" in catalog.columns
    assert "ra" in catalog.columns
    assert "dec" in catalog.columns


def test_from_astropy_qtable():
    a = np.array([1, 4, 5], dtype=np.int32)
    b = [2.0, 5.0, 8.5]
    c = ["x", "y", "z"]
    d = [10, 20, 30] * u.m / u.s
    ra = [2.0, 5.0, 10.0]
    dec = [-30.0, 35.0, 15.0]

    qt = QTable([a, b, c, d, ra, dec], names=("a", "b", "c", "d", "RA", "DEC"), meta={"name": "test table"})
    catalog = lsdb.from_astropy(qt)

    assert catalog is not None
    assert len(catalog) == 3
    assert "a" in catalog.columns
    assert "b" in catalog.columns
    assert "c" in catalog.columns
    assert "d" in catalog.columns
    assert "RA" in catalog.columns
    assert "DEC" in catalog.columns


def test_from_astropy_multidimensional_column():
    data = {
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "spectrum": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    }
    table = Table(data)
    catalog = lsdb.from_astropy(table, ra_column="ra", dec_column="dec")
    assert catalog is not None
    assert len(catalog) == 3
    assert "spectrum" in catalog.columns


def test_from_astropy_multidimensional_ragged_column():
    data = {
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "lc": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
    }
    table = Table(data)
    catalog = lsdb.from_astropy(table, ra_column="ra", dec_column="dec")
    assert catalog is not None
    assert len(catalog) == 3
    assert "lc" in catalog.columns
