import numpy as np
import pyarrow as pa
from nested_pandas.nestedframe.io import from_pyarrow

from lsdb.loaders.dataframe.from_dataframe import from_dataframe


def from_astropy(
    table,
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
    schema=None,
    flatten_tensors: bool = False,
    **kwargs,
):
    """Load a catalog from an Astropy Table.

    Note that this is only suitable for small datasets (< 1million rows and
    < 1GB dataframe in-memory). If you need to deal with large datasets, consider
    using the hats-import package: https://hats-import.readthedocs.io/

    Parameters
    ----------
    table : astropy.table.Table
        The Astropy Table (or QTable).
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
    partition_bytes : int or None, default None
        The desired partition size, in bytes. Only one of
        `partition_rows` or `partition_bytes` should be specified.
    margin_order : int, default -1
        The order at which to generate the margin cache.
    margin_threshold : float or None, default 5
        The threshold (in arcseconds) for including sources in the margin cache. If None, and
        margin_order is specified, the margin cache will include all sources in the margin pixels.
    should_generate_moc : bool, default True
        If True, generates a MOC for the catalog.
    moc_max_order : int, default 10
        The maximum order to use when generating the MOC.
    use_pyarrow_types : bool, default True
        If True, uses PyArrow backed types in the resulting catalog.
    schema : pa.Schema or None, default None
        The arrow schema to create the catalog with. If None, the schema is
        automatically inferred from the DataFrame conversion of the table
        using `pa.Schema.from_pandas`.
    flatten_tensors : bool, default False
        If True, flattens multidimensional columns to 2D arrays in the
        resulting catalog.
    **kwargs
        Additional arguments to pass to the Dataframe loader.

    Returns
    -------
    Catalog
        The loaded catalog.

    Examples
    --------
    >>> from astropy.table import Table
    >>> import lsdb
    >>> data = {
    ...     "ra": [10.0, 20.0, 30.0],
    ...     "dec": [-10.0, -20.0, -30.0],
    ...     "magnitude": [15.0, 16.5, 14.2],
    ... }
    >>> table = Table(data)
    >>> catalog = lsdb.from_astropy(table, ra_column="ra", dec_column="dec")
    >>> catalog.head()
                           ra   dec  magnitude
    _healpix_29                               
    1212933045629049957  10.0 -10.0       15.0
    1176808107119886823  20.0 -20.0       16.5
    2510306432296314470  30.0 -30.0       14.2
    """
    # Go through pyarrow to convert the table to a dataframe.
    # Don't use table.to_pandas() as that would lose multidimensional column support
    arrow_table = _astropy_to_pyarrow_table(table, flatten_tensors=flatten_tensors)
    dataframe = from_pyarrow(arrow_table)

    return from_dataframe(
        dataframe,
        ra_column=ra_column,
        dec_column=dec_column,
        lowest_order=lowest_order,
        highest_order=highest_order,
        drop_empty_siblings=drop_empty_siblings,
        partition_rows=partition_rows,
        partition_bytes=partition_bytes,
        margin_order=margin_order,
        margin_threshold=margin_threshold,
        should_generate_moc=should_generate_moc,
        moc_max_order=moc_max_order,
        use_pyarrow_types=use_pyarrow_types,
        schema=schema,
        **kwargs,
    )


# TODO: Code pulled from hats-import, potentially should move to hats
# In which case, remove this and use the hats version directly
# https://github.com/astronomy-commons/hats-import/issues/623
def _np_to_pyarrow_array(array: np.ndarray, *, flatten_tensors: bool) -> pa.Array:
    """Convert a numpy array to a pyarrow"""
    # We usually have the "wrong" byte order from FITS
    array = np.asanyarray(array, dtype=array.dtype.newbyteorder("="))
    values = pa.array(array.reshape(-1))
    # "Base" type
    if array.ndim == 1:
        return values
    # Flat multidimensional nested values if asked
    if flatten_tensors and array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    pa_list_array = pa.FixedSizeListArray.from_arrays(values, np.prod(array.shape[1:]))
    # An extra dimension is represented as a list array
    if array.ndim == 2:
        return pa_list_array
    # array.ndim > 2
    # Multiple extra dimensions are represented as a tensor array
    tensor_type = pa.fixed_shape_tensor(values.type, shape=array.shape[1:])
    return pa.FixedShapeTensorArray.from_storage(tensor_type, pa_list_array)


def _astropy_to_pyarrow_table(astropy_table, *, flatten_tensors: bool) -> pa.Table:
    """Convert astropy.table.Table to pyarrow.Table"""
    pa_arrays = {}
    for column in astropy_table.columns:
        np_array = np.asarray(astropy_table[column])
        pa_arrays[column] = _np_to_pyarrow_array(np_array, flatten_tensors=flatten_tensors)
    return pa.table(pa_arrays)
