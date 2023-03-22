import lsdb


def test_read_hipscat(small_sky_order1_dir):
    cat = lsdb.read_hipscat(small_sky_order1_dir)
    print("")
    print(cat._ddf.compute())
