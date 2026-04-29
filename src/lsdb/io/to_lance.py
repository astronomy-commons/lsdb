from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm
from upath import UPath

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset

try:
    import lancedb
except ImportError as err:
    raise ImportError(
        "to_lance requires the `lancedb` package. Install it with `pip install lancedb`."
    ) from err

_TABLE_NAME = "data"


def to_lance(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    table_name: str = _TABLE_NAME,
    overwrite: bool = False,
    progress_bar: bool = True,
) -> None:
    """Writes a catalog to a Lance dataset.

    All partitions are written as a single flat Lance dataset. Every column in
    the catalog — including the HEALPix spatial index — is preserved. The
    resulting dataset can be opened with
    ``lancedb.connect(base_catalog_path).open_table("data")``.

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

    base_catalog_path = UPath(base_catalog_path)
    lance_table_path = base_catalog_path / f"{table_name}.lance"
    if lance_table_path.exists() and any(lance_table_path.iterdir()):
        if not overwrite:
            raise ValueError(
                f"A Lance table already exists at '{lance_table_path}'."
                " Choose a different path or set overwrite=True to overwrite the existing dataset."
            )

    path = str(base_catalog_path)
    # pylint: disable=protected-access
    delayed_partitions = catalog._ddf.to_delayed()
    pixel_partition_pairs = list(catalog._ddf_pixel_map.items())

    db = lancedb.connect(path)
    table: lancedb.Table | None = None

    for _, partition_index in tqdm(
        pixel_partition_pairs,
        desc="Writing to Lance",
        disable=not progress_bar,
    ):
        df = delayed_partitions[partition_index].compute()
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

    # TODO: Replace with appropriate logging message and level
    if table is not None:
        print("Optimizing Lance dataset...")
        table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
        print(f"Finished writing output to lance. Path: {path}, Table name: {table_name}")
