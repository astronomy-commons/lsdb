from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm
from upath import UPath

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset

_TABLE_NAME = "data"


def _lance_storage_options_from_upath(path: UPath) -> dict[str, str] | None:
    """Translate a UPath's fsspec-style ``storage_options`` into the key
    names lance/lancedb's Rust ``object_store`` backend expects.

    fsspec filesystems (s3fs, gcsfs, adlfs, ...) and lance's ``object_store``
    each have their own, incompatible ``storage_options`` schemas. A UPath
    built for read/write via fsspec can't be handed to
    ``lancedb.connect(storage_options=...)`` as-is -- this maps the handful
    of fields we care about (endpoint, credentials, TLS) between the two.

    Parameters
    ----------
    path : UPath
        A UPath instance pointing at the target Lance dataset location.

    Returns
    -------
    dict[str, str] | None
        A ``storage_options`` dict suitable for ``lancedb.connect``, or
        ``None`` if the path's protocol needs no special handling (e.g.
        local filesystem paths).
    """
    protocol = path.protocol
    fsso = dict(path.storage_options or {})

    if protocol == "s3":
        return _map_s3_storage_options(fsso)

    # Unknown/unsupported protocol (http, memory, ...) -- nothing sensible
    # to translate; let lance/lancedb use its own defaults.
    return None


def _map_s3_storage_options(fsso: dict) -> dict[str, str]:
    """Translate fsspec-style S3 storage_options into the key names
    lance/lancedb's Rust ``object_store`` backend expects.

    Will read endpoint_url/region_name from the top level of ``fsso``, or
    from ``fsso["client_kwargs"]`` if not present at the top level.

    Https endpoints are passed through as-is. Plain http endpoints are passed
    but accompanied by ``allow_http=true`` to opt into the object_store's
    refusal to use plain http endpoints by default.

    Parameters
    ----------
    fsso : dict
        A dict of fsspec-style S3 storage_options, e.g. from a UPath's
        ``storage_options`` attribute.

    Returns
    -------
    dict[str, str]
        A dict of storage_options suitable for ``lancedb.connect``.
    """
    lance_so: dict[str, str] = {}

    client_kwargs = fsso.get("client_kwargs", {}) or {}

    endpoint_url = fsso.get("endpoint_url") or client_kwargs.get("endpoint_url")
    if endpoint_url:
        lance_so["aws_endpoint"] = endpoint_url
        # object_store refuses plain http endpoints unless told it's ok.
        if endpoint_url.startswith("http://"):
            lance_so["allow_http"] = "true"

    region = fsso.get("region_name") or client_kwargs.get("region_name") or fsso.get("region")
    if region:
        lance_so["aws_region"] = region

    key = fsso.get("key")
    secret = fsso.get("secret")
    token = fsso.get("token")
    if key:
        lance_so["aws_access_key_id"] = key
    if secret:
        lance_so["aws_secret_access_key"] = secret
    if token:
        lance_so["aws_session_token"] = token

    if fsso.get("anon"):
        lance_so["aws_skip_signature"] = "true"

    return lance_so


def to_lance(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    table_name: str = _TABLE_NAME,
    overwrite: bool = False,
    progress_bar: bool = True,
    optimize_dataset: bool = True,
) -> None:
    """Writes a catalog to a Lance dataset.

    All primary catalog partitions are written as a single flat Lance dataset.
    Every column in the catalog — including the HEALPix spatial index — is preserved.
    The margin catalog (if present) is not written to Lance. The resulting dataset
    can be opened with ``lancedb.connect(base_catalog_path).open_table("data")``.

    Parameters
    ----------
    catalog : HealpixDataset
        The catalog to export.
    base_catalog_path : str | Path | UPath
        Path where the Lance dataset will be written.
    table_name : str, default "data"
        Name of the table to create in the Lance database. This is the name used
        to open the table later with lancedb.
    overwrite : bool, default False
        If True, an existing dataset at ``base_catalog_path`` is overwritten.
        If False and a dataset already exists there, an error is raised.
    progress_bar : bool, default True
        If True, shows a progress bar while writing partitions.
    optimize_dataset : bool, default True
        If True, optimizes the Lance dataset after writing all partitions.
        This will improve query performance but will increase the total time required
        to write the dataset.

    Raises
    ------
    ImportError
        If the ``lancedb`` package is not installed.
    ValueError
        If a dataset already exists at ``base_catalog_path`` and ``overwrite=False``.
    RuntimeError
        If the catalog is empty and no data is written.

    Examples
    --------
    Export a catalog and open it with lancedb:

    >>> import lsdb
    >>> catalog = lsdb.read_hats("path/to/small_sky")  # doctest: +SKIP
    >>> catalog.to_lance("/tmp/my_catalog")  # doctest: +SKIP

    Open the result:

    >>> import lancedb  # doctest: +SKIP
    >>> db = lancedb.connect("/tmp/my_catalog")  # doctest: +SKIP
    >>> tbl = db.open_table("data")  # doctest: +SKIP
    """

    try:
        # pylint: disable=import-outside-toplevel
        import lancedb
    except ImportError as err:
        raise ImportError(
            "to_lance requires the `lancedb` package. Install it with `pip install lancedb`."
        ) from err

    base_catalog_path = UPath(base_catalog_path)
    lance_table_path = base_catalog_path / f"{table_name}.lance"
    if lance_table_path.exists() and any(lance_table_path.iterdir()):
        if not overwrite:
            raise ValueError(
                f"A Lance table already exists at '{lance_table_path}'."
                " Choose a different path or set overwrite=True to overwrite the existing dataset."
            )

    path = str(base_catalog_path)
    storage_options = _lance_storage_options_from_upath(base_catalog_path)
    # pylint: disable=protected-access
    delayed_partitions = catalog.to_delayed()

    db = lancedb.connect(path, storage_options=storage_options)
    table: lancedb.Table | None = None

    for _, partition in tqdm(
        enumerate(delayed_partitions),
        total=len(delayed_partitions),
        desc="Writing to Lance",
        disable=not progress_bar,
    ):
        df = partition.compute()
        if len(df) == 0:
            continue

        pa_table = pa.Table.from_pandas(df.reset_index(), preserve_index=False)

        if table is None:
            mode = "overwrite" if overwrite else "create"
            table = db.create_table(table_name, pa_table, mode=mode)
        else:
            table.add(pa_table)

    if table is None:
        raise RuntimeError("The output catalog is empty. No data was written to Lance.")

    if optimize_dataset:
        # TODO: Replace with appropriate logging message and level
        print("Optimizing Lance dataset...")
        table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)

    print(f"Finished writing output to lance. Path: {path}, Table name: {table_name}")
