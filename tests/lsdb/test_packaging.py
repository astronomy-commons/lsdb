import lsdb


def test_lsdb_version():
    """Check to see that we can get the lsdb version"""
    assert lsdb.__version__ is not None
