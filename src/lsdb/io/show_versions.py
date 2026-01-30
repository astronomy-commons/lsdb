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
    """Print runtime versions and system info, useful for bug reports."""
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    maxlen = max(len(x) for x in deps) + 1
    print("\n--------      SYSTEM INFO      --------")
    for k, v in sys_info.items():
        print(f"{k:<{maxlen}}: {v}")
    print("--------   INSTALLED VERSIONS   --------")
    for k, v in deps.items():
        print(f"{k:<{maxlen}}: {v}")
