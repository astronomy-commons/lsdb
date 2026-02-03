from __future__ import annotations

import pandas as pd
import pyarrow as pa

from lsdb.catalog import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader
from lsdb.loaders.dataframe.margin_catalog_generator import MarginCatalogGenerator


# pylint: disable=too-many-arguments
def from_dataframe(
    dataframe: pd.DataFrame,
    *,
    ra_column: str | None = None,
    dec_column: str | None = None,
    lowest_order: int = 0,
    highest_order: int = 7,
    drop_empty_siblings: bool = True,
    partition_rows: int | None = None,
    partition_bytes: int | None = None,
    margin_order: int = -1,
    margin_threshold: float | None = 5.0,
    should_generate_moc: bool = True,
    moc_max_order: int = 10,
    use_pyarrow_types: bool = True,
    schema: pa.Schema | None = None,
    **kwargs,
) -> Catalog:
    """Load a catalog from a Pandas Dataframe.

    Note that this is only suitable for small datasets (< 1million rows and
    < 1GB dataframe in-memory). If you need to deal with large datasets, consider
    using the hats-import package: https://hats-import.readthedocs.io/

    Parameters
    ----------
    dataframe : pd.Dataframe
        The catalog Pandas Dataframe.
    ra_column : str, optional
        The name of the right ascension column. By default,
        case-insensitive versions of 'ra' are detected.
    dec_column : str, optional
        The name of the declination column. By default,
        case-insensitive versions of 'dec' are detected.
    lowest_order : int, default 0
        The lowest partition order. Defaults to 0.
    highest_order : int, default 7
        The highest partition order. Defaults to 7.
    drop_empty_siblings : bool, default True
        When determining final partitionining, if 3 of 4 pixels are empty,
        keep only the non-empty pixel
    partition_rows : int or None, default None
        The desired partition size, in number of rows. Only one of
        `partition_rows` or `partition_bytes` should be specified.

        Note: partitioning is spatial (HEALPix-based). `partition_rows` is a best-effort target,
        and the resulting number of partitions is limited by `highest_order` and the sky footprint
        of your data (e.g., if all rows fall into a single HEALPix pixel at `highest_order`, you will
        still get a single partition).
    partition_bytes : int or None, default None
        The desired partition size, in bytes. Only one of
        `partition_rows` or `partition_bytes` should be specified.

        Note: as with `partition_rows`, this is a best-effort target for spatial (HEALPix-based)
        partitioning and is limited by `highest_order`.
    margin_order : int, default -1
        The order at which to generate the margin cache.
    margin_threshold : float or None, default 5
        The size of the margin cache boundary, in arcseconds. If None, and
        margin order is not specified, the margin cache is not generated. Defaults to 5 arcseconds.
    should_generate_moc : bool, default True
        Should we generate a MOC (multi-order coverage map) of the data.
        It can improve performance when joining/crossmatching to other hats-sharded datasets.
    moc_max_order : int, default 10
        if generating a MOC, what to use as the max order.
    use_pyarrow_types : bool, default True
        If True, the data is backed by pyarrow, otherwise we keep the
        original data types.
    schema : pa.Schema or None
        the arrow schema to create the catalog with. If None, the schema is
        automatically inferred from the provided DataFrame using `pa.Schema.from_pandas`.
    **kwargs :
        Arguments to pass to the creation of the catalog info.

    Returns
    -------
    Catalog
        Catalog object loaded from the given parameters

    Raises
    ------
    ValueError
        If RA/Dec columns are not found or contain NaN values.

    Examples
    --------
    Create a small, synthetic sky catalog and load it into LSDB:

    >>> import lsdb
    >>> from lsdb.nested.datasets import generate_data
    >>> nf = generate_data(1000, 5, seed=0, ra_range=(0.0, 300.0), dec_range=(-50.0, 50.0))
    >>> df = nf.compute()[["ra", "dec", "id"]]
    >>> catalog = lsdb.from_dataframe(df, catalog_name="toy_catalog")
    >>> catalog.head()
                            ra        dec    id
    _healpix_29                                   
    118362963675428450  52.696686  39.675892  8154
    98504457942331510   89.913567  46.147079  3437
    70433374600953220   40.528952  35.350965  8214
    154968715224527848   17.57041    29.8936  9853
    67780378363846894    45.08384   31.95611  8297
    """
    # Load the catalog.
    catalog = DataframeCatalogLoader(
        dataframe,
        ra_column=ra_column,
        dec_column=dec_column,
        lowest_order=lowest_order,
        highest_order=highest_order,
        drop_empty_siblings=drop_empty_siblings,
        partition_rows=partition_rows,
        partition_bytes=partition_bytes,
        should_generate_moc=should_generate_moc,
        moc_max_order=moc_max_order,
        use_pyarrow_types=use_pyarrow_types,
        schema=schema,
        **kwargs,
    ).load_catalog()
    catalog.margin = MarginCatalogGenerator(
        catalog,
        margin_order,
        margin_threshold,
        use_pyarrow_types,
        **kwargs,
    ).create_catalog()
    return catalog
