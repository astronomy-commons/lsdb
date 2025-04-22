# Python 3.9 doesn't support "|" for types
from __future__ import annotations

import dask.dataframe as dd

from .core import NestedFrame


def read_parquet(
    path,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    engine="auto",
    use_nullable_dtypes: bool | None = None,
    dtype_backend=None,
    calculate_divisions=None,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem=None,
    **kwargs,
) -> NestedFrame:
    """
    Read a Parquet file into a Dask DataFrame

    This reads a directory of Parquet data into a Dask.dataframe, one file per
    partition.  It selects the index among the sorted columns if any exist.

    Docstring copied from `dask.dataframe.read_parquet`

    Parameters
    ----------
    path : str or list
        Source directory for data, or path(s) to individual parquet files.
        Prefix with a protocol like ``s3://`` to read from alternative
        filesystems. To read from multiple files you can pass a globstring or a
        list of paths, with the caveat that they must all have the same
        protocol.
    columns : str or list, default None
        Field name(s) to read in as columns in the output. By default all
        non-index fields will be read (as determined by the pandas parquet
        metadata, if present). Provide a single field name instead of a list to
        read in the data as a Series.
    filters : Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]], default None
        List of filters to apply, like ``[[('col1', '==', 0), ...], ...]``.
        Using this argument will result in row-wise filtering of the final partitions.

        Predicates can be expressed in disjunctive normal form (DNF). This means that
        the inner-most tuple describes a single column predicate. These inner predicates
        are combined with an AND conjunction into a larger predicate. The outer-most
        list then combines all of the combined filters with an OR disjunction.

        Predicates can also be expressed as a ``List[Tuple]``. These are evaluated
        as an AND conjunction. To express OR in predicates, one must use the
        (preferred for "pyarrow") ``List[List[Tuple]]`` notation.
    index : str, list or False, default None
        Field name(s) to use as the output frame index. By default will be
        inferred from the pandas parquet file metadata, if present. Use ``False``
        to read all fields as columns.
    categories : list or dict, default None
        For any fields listed here, if the parquet encoding is Dictionary,
        the column will be created with dtype category. Use only if it is
        guaranteed that the column is encoded as dictionary in all row-groups.
        If a list, assumes up to 2**16-1 labels; if a dict, specify the number
        of labels expected; if None, will load categories automatically for
        data written by dask, not otherwise.
    storage_options : dict, default None
        Key/value pairs to be passed on to the file-system backend, if any.
        Note that the default file-system backend can be configured with the
        ``filesystem`` argument, described below.
    open_file_options : dict, default None
        Key/value arguments to be passed along to ``AbstractFileSystem.open``
        when each parquet data file is open for reading. Experimental
        (optimized) "precaching" for remote file systems (e.g. S3, GCS) can
        be enabled by adding ``{"method": "parquet"}`` under the
        ``"precache_options"`` key. Also, a custom file-open function can be
        used (instead of ``AbstractFileSystem.open``), by specifying the
        desired function under the ``"open_file_func"`` key.
    engine : {'auto', 'pyarrow'}
        Parquet library to use. Defaults to 'auto', which uses ``pyarrow`` if
        it is installed, and falls back to the deprecated ``fastparquet`` otherwise.
        Note that ``fastparquet`` does not support all functionality offered by
        ``pyarrow``.
        This is also used by third-party packages (e.g. CuDF) to inject bespoke engines.
    use_nullable_dtypes : {False, True}
        Whether to use extension dtypes for the resulting ``DataFrame``.

        .. note::

            This option is deprecated. Use "dtype_backend" instead.

    dtype_backend : {'numpy_nullable', 'pyarrow'}, defaults to NumPy backed DataFrames
        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,
        nullable dtypes are used for all dtypes that have a nullable implementation
        when 'numpy_nullable' is set, pyarrow is used for all dtypes if 'pyarrow'
        is set.
        ``dtype_backend="pyarrow"`` requires ``pandas`` 1.5+.
    calculate_divisions : bool, default False
        Whether to use min/max statistics from the footer metadata (or global
        ``_metadata`` file) to calculate divisions for the output DataFrame
        collection. Divisions will not be calculated if statistics are missing.
        This option will be ignored if ``index`` is not specified and there is
        no physical index column specified in the custom "pandas" Parquet
        metadata. Note that ``calculate_divisions=True`` may be extremely slow
        when no global ``_metadata`` file is present, especially when reading
        from remote storage. Set this to ``True`` only when known divisions
        are needed for your workload (see :ref:`dataframe-design-partitions`).
    ignore_metadata_file : bool, default False
        Whether to ignore the global ``_metadata`` file (when one is present).
        If ``True``, or if the global ``_metadata`` file is missing, the parquet
        metadata may be gathered and processed in parallel. Parallel metadata
        processing is currently supported for ``ArrowDatasetEngine`` only.
    metadata_task_size : int, default configurable
        If parquet metadata is processed in parallel (see ``ignore_metadata_file``
        description above), this argument can be used to specify the number of
        dataset files to be processed by each task in the Dask graph.  If this
        argument is set to ``0``, parallel metadata processing will be disabled.
        The default values for local and remote filesystems can be specified
        with the "metadata-task-size-local" and "metadata-task-size-remote"
        config fields, respectively (see "dataframe.parquet").
    split_row_groups : 'infer', 'adaptive', bool, or int, default 'infer'
        If True, then each output dataframe partition will correspond to a single
        parquet-file row-group. If False, each partition will correspond to a
        complete file.  If a positive integer value is given, each dataframe
        partition will correspond to that number of parquet row-groups (or fewer).
        If 'adaptive', the metadata of each file will be used to ensure that every
        partition satisfies ``blocksize``. If 'infer' (the default), the
        uncompressed storage-size metadata in the first file will be used to
        automatically set ``split_row_groups`` to either 'adaptive' or ``False``.
    blocksize : int or str, default 'default'
        The desired size of each output ``DataFrame`` partition in terms of total
        (uncompressed) parquet storage space. This argument is currently used to
        set the default value of ``split_row_groups`` (using row-group metadata
        from a single file), and will be ignored if ``split_row_groups`` is not
        set to 'infer' or 'adaptive'. Default is 256 MiB.
    aggregate_files : bool or str, default None
        WARNING: Passing a string argument to ``aggregate_files`` will result
        in experimental behavior. This behavior may change in the future.

        Whether distinct file paths may be aggregated into the same output
        partition. This parameter is only used when `split_row_groups` is set to
        'infer', 'adaptive' or to an integer >1. A setting of True means that any
        two file paths may be aggregated into the same output partition, while
        False means that inter-file aggregation is prohibited.

        For "hive-partitioned" datasets, a "partition"-column name can also be
        specified. In this case, we allow the aggregation of any two files
        sharing a file path up to, and including, the corresponding directory name.
        For example, if ``aggregate_files`` is set to ``"section"`` for the
        directory structure below, ``03.parquet`` and ``04.parquet`` may be
        aggregated together, but ``01.parquet`` and ``02.parquet`` cannot be.
        If, however, ``aggregate_files`` is set to ``"region"``, ``01.parquet``
        may be aggregated with ``02.parquet``, and ``03.parquet`` may be aggregated
        with ``04.parquet``::

            dataset-path/
            ├── region=1/
            │   ├── section=a/
            │   │   └── 01.parquet
            │   ├── section=b/
            │   └── └── 02.parquet
            └── region=2/
                ├── section=a/
                │   ├── 03.parquet
                └── └── 04.parquet

        Note that the default behavior of ``aggregate_files`` is ``False``.
    parquet_file_extension: str, tuple[str], or None, default (".parq", ".parquet", ".pq")
        A file extension or an iterable of extensions to use when discovering
        parquet files in a directory. Files that don't match these extensions
        will be ignored. This argument only applies when ``paths`` corresponds
        to a directory and no ``_metadata`` file is present (or
        ``ignore_metadata_file=True``). Passing in ``parquet_file_extension=None``
        will treat all files in the directory as parquet files.

        The purpose of this argument is to ensure that the engine will ignore
        unsupported metadata files (like Spark's '_SUCCESS' and 'crc' files).
        It may be necessary to change this argument if the data files in your
        parquet dataset do not end in ".parq", ".parquet", or ".pq".
    filesystem: "fsspec", "arrow", or fsspec.AbstractFileSystem backend to use.
        Specifies the backend to use.
    dataset: dict, default None
        Dictionary of options to use when creating a ``pyarrow.dataset.Dataset`` object.
        These options may include a "filesystem" key to configure the desired
        file-system backend. However, the top-level ``filesystem`` argument will always
        take precedence.

        **Note**: The ``dataset`` options may include a "partitioning" key.
        However, since ``pyarrow.dataset.Partitioning``
        objects cannot be serialized, the value can be a dict of key-word
        arguments for the ``pyarrow.dataset.partitioning`` API
        (e.g. ``dataset={"partitioning": {"flavor": "hive", "schema": ...}}``).
        Note that partitioned columns will not be converted to categorical
        dtypes when a custom partitioning schema is specified in this way.
    read: dict, default None
        Dictionary of options to pass through to ``engine.read_partitions``
        using the ``read`` key-word argument.
    arrow_to_pandas: dict, default None
        Dictionary of options to use when converting from ``pyarrow.Table`` to
        a pandas ``DataFrame`` object. Only used by the "arrow" engine.
    **kwargs: dict (of dicts)
        Options to pass through to ``engine.read_partitions`` as stand-alone
        key-word arguments. Note that these options will be ignored by the
        engines defined in ``dask.dataframe``, but may be used by other custom
        implementations.

    Examples
    --------
    >>> df = dd.read_parquet('s3://bucket/my-parquet-data')  # doctest: +SKIP
    """
    return NestedFrame.from_dask_dataframe(
        dd.read_parquet(
            path=path,
            columns=columns,
            filters=filters,
            categories=categories,
            index=index,
            storage_options=storage_options,
            engine=engine,
            use_nullable_dtypes=use_nullable_dtypes,
            dtype_backend=dtype_backend,
            calculate_divisions=calculate_divisions,
            ignore_metadata_file=ignore_metadata_file,
            metadata_task_size=metadata_task_size,
            split_row_groups=split_row_groups,
            blocksize=blocksize,
            aggregate_files=aggregate_files,
            parquet_file_extension=parquet_file_extension,
            filesystem=filesystem,
            **kwargs,
        )
    )
