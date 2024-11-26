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
    ra_column: str = "ra",
    dec_column: str = "dec",
    lowest_order: int = 0,
    highest_order: int = 7,
    drop_empty_siblings: bool = False,
    partition_size: int | None = None,
    threshold: int | None = None,
    margin_order: int = -1,
    margin_threshold: float | None = 5.0,
    should_generate_moc: bool = True,
    moc_max_order: int = 10,
    use_pyarrow_types: bool = True,
    schema: pa.Schema | None = None,
    **kwargs,
) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Note that this is only suitable for small datasets (< 1million rows and
    < 1GB dataframe in-memory). If you need to deal with large datasets, consider
    using the hats-import package: https://hats-import.readthedocs.io/

    Args:
        dataframe (pd.Dataframe): The catalog Pandas Dataframe.
        ra_column (str): The name of the right ascension column. Defaults to ra.
        dec_column (str): The name of the declination column. Defaults to dec.
        lowest_order (int): The lowest partition order. Defaults to 0.
        highest_order (int): The highest partition order. Defaults to 7.
        drop_empty_siblings (bool): When determining final partitionining,
            if 3 of 4 pixels are empty, keep only the non-empty pixel
        partition_size (int): The desired partition size, in number of bytes in-memory.
        threshold (int): The maximum number of data points per pixel.
        margin_order (int): The order at which to generate the margin cache.
        margin_threshold (float): The size of the margin cache boundary, in arcseconds. If None, and
            margin order is not specified, the margin cache is not generated. Defaults to 5 arcseconds.
        should_generate_moc (bool): should we generate a MOC (multi-order coverage map)
            of the data. can improve performance when joining/crossmatching to
            other hats-sharded datasets.
        moc_max_order (int): if generating a MOC, what to use as the max order. Defaults to 10.
        use_pyarrow_types (bool): If True, the data is backed by pyarrow, otherwise we keep the
            original data types. Defaults to True.
        schema (pa.Schema): the arrow schema to create the catalog with. If None, the schema is
            automatically inferred from the provided DataFrame using `pa.Schema.from_pandas`.
        **kwargs: Arguments to pass to the creation of the catalog info.

    Returns:
        Catalog object loaded from the given parameters
    """
    catalog = DataframeCatalogLoader(
        dataframe,
        ra_column=ra_column,
        dec_column=dec_column,
        lowest_order=lowest_order,
        highest_order=highest_order,
        drop_empty_siblings=drop_empty_siblings,
        partition_size=partition_size,
        threshold=threshold,
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
