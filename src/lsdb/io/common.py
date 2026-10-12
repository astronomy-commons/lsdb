import math as m
from importlib.metadata import version
from pathlib import Path

import nested_pandas as npd
import pyarrow.parquet as pq
from hats.catalog import TableProperties
from hats.io.file_io.file_io import get_parquet_write_table_kwargs
from hats.pixel_math.spatial_index import split_to_row_groups
from upath import UPath


def new_provenance_properties(
    path: str | Path | UPath | None = None, inherit_provenance: bool = False, **kwargs
) -> dict:
    """Create a new provenance properties dictionary for the dataset.

    Parameters
    ----------
    path: str | Path | UPath | None, default None
        The path to the catalog.
    inherit_provenance : bool, default False
        If the original catalog had some provenance info (like creator or bib references),
        should this new catalog retain all of it?
    **kwargs
        Additional provenance properties.

    Returns
    -------
    dict
        A new provenance dictionary.
    """
    if not inherit_provenance:
        kwargs = {
            "hats_creator": None,
            "bib_reference": None,
            "bib_reference_url": None,
            "creator_did": None,
            "publisher_id": None,
        } | kwargs

    kwargs = {
        "hats_cols_sort": None,
        "hats_cols_survey_id": None,
    } | kwargs
    return TableProperties.new_provenance_dict(path, builder=f"lsdb v{version('lsdb')}", **kwargs)


def round_sig(value: float, digits: int = 5) -> float:
    """Round a float to a number of significant figures, keeping it a float.

    Equivalent to ``float(f"{value:.{digits}g}")``, but computes the rounding
    numerically instead of round-tripping through a string.

    Parameters
    ----------
    value : float
        The value to round.
    digits : int, default 5
        The number of significant figures to keep.

    Returns
    -------
    float
        The value rounded to the given number of significant figures.
    """
    if value == 0:
        return 0.0
    decimal_places = digits - 1 - m.floor(m.log10(abs(value)))
    return round(value, decimal_places)


def write_partition_parquet(
    df: npd.NestedFrame,
    pixel_path: UPath,
    hp_pixel_order: int | None = None,
    row_group_kwargs: dict | None = None,
    **kwargs,
):
    """Write a partition to a parquet file with the HATS default write settings.

    Parameters
    ----------
    df : npd.NestedFrame
        Partition to write
    pixel_path : UPath
        Location of the parquet file
    hp_pixel_order : int or None, default None
        HealPix order of this pixel. Used for calculating row groups with subtile_order_delta.
    row_group_kwargs : dict or None, default None
        kwargs for splitting this partition into row groups, as in
        hats.pixel_math.spatial_index.split_to_row_groups().
    **kwargs
        Arguments to pass to ``pyarrow.parquet.write_table``, taking precedence over
        the defaults from ``hats.io.file_io.file_io.get_parquet_write_table_kwargs``.
        ``list_struct`` and ``large_list`` are passed to ``NestedFrame.to_pyarrow`` instead.
    """
    table = df.to_pyarrow(
        list_struct=kwargs.pop("list_struct", False), large_list=kwargs.pop("large_list", False)
    )
    rowgroup_tables = split_to_row_groups(table, row_group_kwargs, hp_pixel_order)
    write_table_kwargs = get_parquet_write_table_kwargs(table.schema, write_table_kwargs=kwargs)

    # remove 'row_group_size' as it is an invalid kwarg for pq.ParquetWriter
    del write_table_kwargs["row_group_size"]

    # write all tables as row groups
    with pq.ParquetWriter(
        pixel_path.path,
        table.schema,
        filesystem=pixel_path.fs,
        **write_table_kwargs,
    ) as writer:
        for rg_table in rowgroup_tables:
            writer.write_table(rg_table)
