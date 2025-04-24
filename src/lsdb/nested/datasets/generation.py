from nested_pandas import datasets

import lsdb.nested as nd


def generate_data(n_base, n_layer, npartitions=1, seed=None) -> nd.NestedFrame:
    """Generates a toy dataset.

    Docstring copied from nested-pandas.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    npartitions: int
        The number of partitions to split the data into.
    seed : int
        A seed to use for random generation of data

    Returns
    -------
    NestedFrame
        The constructed Dask NestedFrame.

    Examples
    --------
    >>> import lsdb.nested as nd
    >>> nd.datasets.generate_data(10,100)
    >>> nd.datasets.generate_data(10, {"nested_a": 100, "nested_b": 200})
    """

    # Use nested-pandas generator
    base_nf = datasets.generate_data(n_base, n_layer, seed=seed)

    # Convert to nested-dask
    base_nf = nd.NestedFrame.from_pandas(base_nf).repartition(npartitions=npartitions)

    return base_nf


def generate_parquet_file(n_base, n_layer, path, file_per_layer=True, npartitions=1, seed=None):
    """Generates a toy dataset and outputs it to one or more parquet files.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    path : str,
        The path to the parquet file to write to if `file_per_layer` is `False`,
        and otherwise the path to the directory to write the parquet file for
        each layer.
    file_per_layer : bool, default=True
        TODO: Currently only True is supported.
        If True, write each layer to its own parquet file. Otherwise, write
        the generated to a single parquet file representing a nested dataset.
    npartitions : int, default=1
        The number of Dask partitions to split the generated data into for each layer.
    seed : int, default=None
        A seed to use for random generation of data

    Returns
    -------
    None
    """
    nf = generate_data(n_base, n_layer, npartitions, seed)
    nf.to_parquet(path, by_layer=file_per_layer, write_index=False)
