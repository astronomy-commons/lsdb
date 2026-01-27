import lsdb


def test_show_versions(capsys):
    lsdb.show_versions()
    captured = capsys.readouterr().out
    assert captured.startswith("\n--------      SYSTEM INFO      --------")
    assert "lsdb" in captured
    assert "hats" in captured
    assert "nested-pandas" in captured
    assert "pyarrow" in captured
    assert "fsspec" in captured
