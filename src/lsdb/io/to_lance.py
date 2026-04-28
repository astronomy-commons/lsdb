from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm
from upath import UPath

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def to_lance(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    overwrite: bool = False,
    progress_bar: bool = True,
) -> None:
    """Writes a catalog to a Lance dataset.

    All partitions are written as a single flat Lance dataset. Every column in
    the catalog — including the HEALPix spatial index — is preserved. The
    resulting dataset can be opened with ``lance.dataset(base_catalog_path)``.

    Parameters
    ----------
    catalog : HealpixDataset
        The catalog to export.
    base_catalog_path : str | Path | UPath
        Path where the Lance dataset will be written.
    overwrite : bool, default False
        If True, an existing dataset at ``base_catalog_path`` is overwritten.
        If False and a dataset already exists there, an error is raised.
    progress_bar : bool, default True
        If True, shows a progress bar while writing partitions.

    Raises
    ------
    ImportError
        If the ``lance`` package is not installed.
    RuntimeError
        If the catalog is empty and no data is written.

    Examples
    --------
    Export a catalog and open it with lance:

    >>> import lsdb
    >>> catalog = lsdb.read_hats("path/to/small_sky")  # doctest: +SKIP
    >>> catalog.to_lance("/tmp/my_catalog.lance")  # doctest: +SKIP

    Open the result:

    >>> import lance  # doctest: +SKIP
    >>> ds = lance.dataset("/tmp/my_catalog.lance")  # doctest: +SKIP
    """
    try:
        import lance
    except ImportError as err:
        raise ImportError(
            "to_lance requires the `lance` package. Install it with `pip install lance`."
        ) from err

    path = str(base_catalog_path)
    delayed_partitions = catalog._ddf.to_delayed()
    pixel_partition_pairs = list(catalog._ddf_pixel_map.items())

    wrote_any = False

    for pixel, partition_index in tqdm(
        pixel_partition_pairs,
        desc="Writing to Lance",
        disable=not progress_bar,
    ):
        df = delayed_partitions[partition_index].compute()
        if len(df) == 0:
            continue

        pa_table = pa.Table.from_pandas(df, preserve_index=True)

        if not wrote_any:
            mode = "overwrite" if overwrite else "create"
            lance.write_dataset(pa_table, path, mode=mode)
            wrote_any = True
        else:
            lance.write_dataset(pa_table, path, mode="append")

    if not wrote_any:
        raise RuntimeError("The output catalog is empty. No data was written to Lance.")

    ds = lance.dataset(path)
    ds.optimize.compact_files()
    ds.cleanup_old_versions(older_than=timedelta(0), delete_unverified=True)
