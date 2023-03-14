import lsdb


def test_from_hipscat(small_sky_order1_dir):
    cat = lsdb.from_hipscat(small_sky_order1_dir)
    print("")
    print(cat._ddf.compute())
