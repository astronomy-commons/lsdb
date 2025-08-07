from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask
import hats
import hats.io as file_io
import nested_pandas as npd
import numpy as np
from hats.catalog import CatalogType, PartitionInfo, TableProperties
from hats.catalog.catalog_collection import CatalogCollection
from hats.pixel_math import HealpixPixel
from upath import UPath

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


@dask.delayed
def perform_write(
    df: npd.NestedFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: str | Path | UPath,
    separation_column: str | None = None,
    **kwargs,
) -> tuple[int, float]:
    """Writes a pandas dataframe to a single parquet file and returns the total count
    for the partition as well as a count histogram at the specified order.

    Args:
        df (npd.NestedFrame): dataframe to write to file
        hp_pixel (HealpixPixel): HEALPix pixel of file to be written
        base_catalog_dir (path-like): Location of the base catalog directory to write to
        separation_column (str): The name of the crossmatch separation column
        **kwargs: other kwargs to pass to pq.write_table method

    Returns:
        The total number of points on the partition and the maximum separation between
        any two of its points. It returns a maximum separation of -1 if a separation
        column is not provided.
    """
    if len(df) == 0:
        return (0, -1)
    pixel_dir = file_io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    file_io.file_io.make_directory(pixel_dir, exist_ok=True)
    pixel_path = file_io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel)
    df.to_parquet(pixel_path.path, filesystem=pixel_path.fs, **kwargs)
    max_sep = df[separation_column].max() if separation_column is not None else -1
    return len(df), max_sep


# pylint: disable=protected-access,too-many-locals
def to_association(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    catalog_name: str | None = None,
    primary_catalog_dir: str | Path | UPath | None = None,
    primary_column_association: str | None = None,
    primary_id_column: str | None = None,
    join_catalog_dir: str | Path | UPath | None = None,
    join_column_association: str | None = None,
    join_to_primary_id_column: str | None = None,
    join_id_column: str | None = None,
    separation_column: str | None = None,
    overwrite: bool = False,
    **kwargs,
):
    """Writes a crossmatching product to disk, in HATS association table format.
    The output catalog comprises partition parquet files and respective metadata.

    The column name arguments should reflect the column names on the corresponding
    primary and join OBJECT catalogs, so that the association table can be used
    to perform equijoins on the two sides and recreate the crossmatch.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_path (str): Location where catalog is saved to
        catalog_name (str): The name of the output catalog
        primary_catalog_dir (path-like): The path to the primary catalog
        primary_column_association (str): The column in the association catalog
            that matches the primary (left) side of join
        primary_id_column (str): The id column in the primary catalog
        join_catalog_dir (path-like): The path to the join catalog
        join_column_association (str): The column in the association catalog
            that matches the joining (right) side of join
        join_id_column (str): The id column in the join catalog
        separation_column (str): The name of the crossmatch separation column
        overwrite (bool): If True existing catalog is overwritten
        **kwargs: Arguments to pass to the parquet write operations

    Notes:
        To configure the appropriate column names, consider two tables that do not
        share an identifier space (e.g. two surveys), and the way you could go about
        joining them together with an association table::

            TABLE GAIA_SOURCE {
                DESIGNATION <primary key>
            }

            TABLE SDSS {
                SDSS_ID <primary key>
            }

        And a SQL query to join them with as association table would look like::

            SELECT g.DESIGNATION as gaia_id, s.SDSS_ID as sdss_id
            FROM GAIA_SOURCE g
            JOIN association_table a
                ON a.primary_id_column = g.DESIGNATION
            JOIN SDSS s
                ON a.join_id_column = s.SDSS_ID

        Consider instead an object table, joining to a detection table::

            TABLE OBJECT {
                ID <primary key>
            }

            TABLE DETECTION {
                DETECTION_ID <primary key>
                OBJECT_ID <foreign key>
            }

        And a SQL query to join them would look like::

            SELECT o.ID as object_id, d.DETECTION_ID as detection_id
            FROM OBJECT o
            JOIN DETECTION d
                ON o.ID = d.OBJECT_ID

        This is important, as there are three different column names, but really only
        two meaningful identifiers. For this example, the arguments for this method would
        be as follows::

            primary_id_column = "ID",
            join_to_primary_id_column = "OBJECT_ID",
            join_id_column = "DETECTION_ID",
    """
    column_args = _check_catalogs_and_columns(
        catalog.columns,
        primary_catalog_dir=primary_catalog_dir,
        primary_column_association=primary_column_association,
        primary_id_column=primary_id_column,
        join_catalog_dir=join_catalog_dir,
        join_column_association=join_column_association,
        join_to_primary_id_column=join_to_primary_id_column,
        join_id_column=join_id_column,
        separation_column=separation_column,
    )

    # Create the output directory for the catalog
    base_catalog_path = file_io.file_io.get_upath(base_catalog_path)
    if file_io.file_io.directory_has_contents(base_catalog_path):
        if not overwrite:
            raise ValueError(
                f"base_catalog_path ({str(base_catalog_path)}) contains files."
                " choose a different directory or set overwrite to True."
            )
        file_io.file_io.remove_directory(base_catalog_path)
    file_io.file_io.make_directory(base_catalog_path, exist_ok=True)

    # Save partition parquet files
    pixels, counts, max_separations = write_partitions(
        catalog, base_catalog_dir_fp=base_catalog_path, separation_column=separation_column, **kwargs
    )

    # Save parquet metadata
    file_io.write_parquet_metadata(base_catalog_path, create_thumbnail=False)

    # Save partition info
    partition_info = PartitionInfo(pixels)
    partition_info.write_to_file(base_catalog_path / "partition_info.csv")

    # Save catalog info
    info = {
        "catalog_name": catalog_name,
        "catalog_type": CatalogType.ASSOCIATION,
        "contains_leaf_files": True,
        "hats_order": partition_info.get_highest_order(),
        "total_rows": int(np.sum(counts)),
        "moc_sky_fraction": f"{partition_info.calculate_fractional_coverage():0.5f}",
    }

    max_separation = np.max(max_separations)
    if max_separation != -1:
        info = info | {"assn_max_separation": f"{max_separation:0.5f}"}
    info = info | column_args | kwargs

    new_hc_structure = TableProperties(**info)
    new_hc_structure.to_properties_file(base_catalog_path)


def write_partitions(
    catalog: HealpixDataset,
    base_catalog_dir_fp: str | Path | UPath,
    separation_column: str | None,
    **kwargs,
) -> tuple[list[HealpixPixel], list[int], list[float]]:
    """Saves catalog partitions as parquet to disk and computes the sparse
    count histogram for each partition. The histogram is either of order 8
    or the maximum pixel order in the catalog, whichever is greater.

    Args:
        catalog (HealpixDataset): A catalog to export
        base_catalog_dir_fp (path-like): Path to the base directory of the catalog
        **kwargs: Arguments to pass to the parquet write operations

    Returns:
        A tuple with the array of non-empty pixels, the array with the total counts
        as well as the array with the maximum point separations.
    """
    results, pixels = [], []
    partitions = catalog._ddf.to_delayed()

    for pixel, partition_index in catalog._ddf_pixel_map.items():
        results.append(
            perform_write(
                partitions[partition_index],
                pixel,
                base_catalog_dir_fp,
                separation_column,
                **kwargs,
            )
        )
        pixels.append(pixel)

    results = dask.compute(*results)
    counts, max_separations = list(zip(*results))

    non_empty_indices = np.nonzero(counts)
    non_empty_pixels = np.array(pixels)[non_empty_indices]
    non_empty_counts = np.array(counts)[non_empty_indices]
    non_empty_max_separations = np.array(max_separations)[non_empty_indices]

    # Check that the catalog is not empty
    if len(non_empty_pixels) == 0:
        raise RuntimeError("The output catalog is empty")

    return list(non_empty_pixels), list(non_empty_counts), list(non_empty_max_separations)


def _check_catalogs_and_columns(
    catalog_columns,
    primary_catalog_dir: str | Path | UPath | None = None,
    primary_column_association: str | None = None,
    primary_id_column: str | None = None,
    join_catalog_dir: str | Path | UPath | None = None,
    join_column_association: str | None = None,
    join_to_primary_id_column: str | None = None,
    join_id_column: str | None = None,
    separation_column: str | None = None,
):
    """Helper function to perform validation of user-inputted catalog and column arguments.

    Returns:
        dictionary to be used in creation of TableProperties
    """
    # Verify that the association columns are present.
    if not primary_column_association:
        raise ValueError("primary_column_association is required")
    if not join_column_association:
        raise ValueError("join_column_association is required")
    if primary_column_association not in catalog_columns:
        raise ValueError("primary_column_association must be a column in input catalog")
    if join_column_association not in catalog_columns:
        raise ValueError("join_column_association must be a column in input catalog")
    if separation_column is not None and separation_column not in catalog_columns:
        raise ValueError("separation_column must be a column in input catalog")

    # Verify that the primary and join catalogs exist, and have the indicated columns.
    if not primary_catalog_dir:
        raise ValueError("primary_catalog_dir is required")
    if not primary_id_column:
        raise ValueError("primary_id_column is required")

    primary_catalog = hats.read_hats(primary_catalog_dir)
    if isinstance(primary_catalog, CatalogCollection):
        primary_catalog = primary_catalog.main_catalog
    if primary_catalog.original_schema and primary_id_column not in primary_catalog.original_schema.names:
        raise ValueError("primary_id_column must be a column in the primary catalog")

    if not join_catalog_dir:
        raise ValueError("join_catalog_dir is required")
    if not join_id_column:
        raise ValueError("join_id_column is required")

    join_catalog = hats.read_hats(join_catalog_dir)
    if isinstance(join_catalog, CatalogCollection):
        join_catalog = join_catalog.main_catalog
    if join_catalog.original_schema:
        if join_id_column not in join_catalog.original_schema.names:
            raise ValueError("join_id_column must be a column in the join catalog")
        if join_to_primary_id_column and join_to_primary_id_column not in join_catalog.original_schema.names:
            raise ValueError("join_to_primary_id_column must be a column in the primary catalog")
    info = {
        "primary_column": primary_id_column,
        "primary_column_association": primary_column_association,
        "primary_catalog": str(primary_catalog_dir),
        "join_column": join_id_column,
        "join_column_association": join_column_association,
        "join_catalog": str(join_catalog_dir),
    }
    if join_to_primary_id_column:
        info["join_to_primary_id_column"] = join_to_primary_id_column
    return info
