import importlib
import os
import platform
import struct
import sys


def _get_sys_info() -> dict[str, str]:
    uname_result = platform.uname()
    return {
        "python": platform.python_version(),
        "python-bits": str(struct.calcsize("P") * 8),
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL") or "",
        "LANG": os.environ.get("LANG") or "",
    }


def _get_dependency_info() -> dict[str, str]:
    deps = [
        "lsdb",
        "hats",
        "nested-pandas",
        "pandas",
        "numpy",
        "dask",
        "pyarrow",
        "fsspec",
    ]

    result: dict[str, str] = {}
    for modname in deps:
        try:
            result[modname] = importlib.metadata.version(modname)
        except Exception:  # pylint: disable=broad-exception-caught # pragma: no cover
            result[modname] = "N/A"
    return result


def show_versions():
    """Print runtime versions and system info, useful for bug reports.

    This reports installed *software* versions (lsdb, hats, their dependencies, and the
    host system). For the *data* versions of loaded catalogs, see
    :func:`show_catalog_versions`.
    """
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    maxlen = max(len(x) for x in deps) + 1
    print("\n--------      SYSTEM INFO      --------")
    for k, v in sys_info.items():
        print(f"{k:<{maxlen}}: {v}")
    print("--------   INSTALLED VERSIONS   --------")
    for k, v in deps.items():
        print(f"{k:<{maxlen}}: {v}")


def _collect_versions(catalogs) -> tuple[dict[str, str | None], list[str]]:
    """Sort catalogs into the collection versions and standalone catalogs they represent.

    Returns a ``{collection_name: collection_version}`` mapping (deduplicated by
    collection, insertion-ordered) and an ordered list of the names of any standalone
    catalogs that have no parent collection. Uses duck typing so that any object exposing
    ``name`` and ``hc_collection`` works.
    """
    collection_versions: dict[str, str | None] = {}
    standalone_names: list[str] = []
    for catalog in catalogs:
        collection = getattr(catalog, "hc_collection", None)
        if collection is None:
            if catalog.name not in standalone_names:
                standalone_names.append(catalog.name)
        else:
            name = collection.collection_properties.name
            if name not in collection_versions:
                collection_versions[name] = collection.collection_version
    return collection_versions, standalone_names


def _format_catalog_versions(*catalogs) -> str:
    """Build the display string for :func:`show_catalog_versions` (see it for details)."""
    if len(catalogs) == 0:
        return "No catalogs provided."

    collection_versions, standalone_names = _collect_versions(catalogs)

    # Align the value column across every line, mirroring show_versions' padded layout.
    labels = [f"Collection: {name}" for name in collection_versions] + standalone_names
    label_w = max(len(label) for label in labels)
    gap = "   "

    lines = []
    for name, version in collection_versions.items():
        label = f"Collection: {name}"
        # An em dash marks a collection that exists but declares no version.
        lines.append(f"{label:<{label_w}}{gap}{version if version is not None else '—'}")
    for name in standalone_names:
        lines.append(f"{name:<{label_w}}{gap}(standalone catalog, no collection version)")
    return "\n".join(lines)


def show_catalog_versions(*catalogs):
    """Print the *data* versions of loaded catalogs (their collections' ``collection_version``).

    This is the data-provenance sibling of :func:`show_versions`: where ``show_versions``
    reports installed *software* versions for bug reports, this reports which data release
    of the data you have loaded. A data version is a property of a collection as a whole,
    so one line is printed per collection (deduplicated when several loaded catalogs share
    one). Collections that declare no version render with an em dash, and standalone
    catalogs with no parent collection are noted as having no collection version.

    Parameters
    ----------
    *catalogs : Catalog
        One or more loaded LSDB catalogs to report on.

    Examples
    --------
    >>> import lsdb
    >>> gaia = lsdb.open_catalog("./gaia_dr3")  # doctest: +SKIP
    >>> lsdb.show_catalog_versions(gaia)  # doctest: +SKIP
    Collection: gaia_dr3   v2.1.0
    """
    print(_format_catalog_versions(*catalogs))
