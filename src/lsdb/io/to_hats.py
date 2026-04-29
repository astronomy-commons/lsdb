from __future__ import annotations

import re
from copy import copy
from pathlib import Path
from typing import Any

import dask
import hats as hc
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hats.catalog import CatalogType, PartitionInfo
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.io import file_io
from hats.io.skymap import write_skymap
from hats.pixel_math import HealpixPixel, spatial_index_to_healpix
from hats.pixel_math.sparse_histogram import HistogramAggregator, SparseHistogram
from tqdm.dask import TqdmCallback
from upath import UPath

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.io.common import new_provenance_properties, set_default_write_table_kwargs

DONE_DIR_NAME = "done"
HISTOGRAM_DIR_NAME = "hists"

WRITE_RESULT_META = pd.DataFrame({"count": pd.Series(dtype=int), "histogram": pd.Series(dtype=object)})


def perform_write(
    df: npd.NestedFrame,
    hp_pixel: HealpixPixel,
    base_catalog_dir: UPath,
    histogram_order: int,
    **kwargs,
) -> pd.DataFrame:
    """Writes a pandas dataframe to a single parquet file and returns the total count
    for the partition as well as a count histogram at the specified order.

    Parameters
    ----------
    df : npd.NestedFrame
        Dataframe to write to file
    hp_pixel : HealpixPixel
        HEALPix pixel of file to be written
    base_catalog_dir : path-like
        Location of the base catalog directory to write to
    histogram_order : int
        Order of the count histogram
    **kwargs
        Other kwargs to pass to pq.write_table method

    Returns
    -------
    tuple[int, SparseHistogram]
        The total number of points on the partition and the sparse count histogram
        at the specified order.
    """
    if len(df) == 0:
        histogram = SparseHistogram([], [], histogram_order)
        write_histogram(histogram, base_catalog_dir, hp_pixel)
        write_done_pixel(base_catalog_dir, hp_pixel)
        return pd.DataFrame({"count": [0], "histogram": [histogram]})
    pixel_dir = hc.io.pixel_directory(base_catalog_dir, hp_pixel.order, hp_pixel.pixel)
    hc.io.file_io.make_directory(pixel_dir, exist_ok=True)
    pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir, hp_pixel)
    df.to_parquet(pixel_path.path, filesystem=pixel_path.fs, **kwargs)
    histogram = calculate_histogram(df, histogram_order)
    write_histogram(histogram, base_catalog_dir, hp_pixel)
    write_done_pixel(base_catalog_dir, hp_pixel)
    return pd.DataFrame({"count": [len(df)], "histogram": [histogram]})


def calculate_histogram(df: npd.NestedFrame, histogram_order: int) -> SparseHistogram:
    """Splits a partition into pixels at a specified order and computes
    the sparse histogram with the respective counts.

    Parameters
    ----------
    df : npd.NestedFrame
        Partition data frame
    histogram_order : int
        Order of the count histogram

    Returns
    -------
    SparseHistogram
        The sparse count histogram for the partition, at the specified order.
    """
    order_pixels = spatial_index_to_healpix(df.index.to_numpy(), target_order=histogram_order)
    gb = df.groupby(order_pixels, sort=False).apply(len)
    indexes, counts_at_indexes = gb.index.to_numpy(), gb.to_numpy(na_value=0)
    return SparseHistogram(indexes, counts_at_indexes, histogram_order)


def _read_pixels_from_files(path: UPath, glob_pattern: str, regex_pattern: str) -> list[HealpixPixel]:
    """Helper function to read HEALPix pixels from files in a directory from a glob pattern and regex pattern.

    Parameters
    ----------
    path : path-like
        Location of the directory to read from
    glob_pattern : str
        Glob pattern to find relevant files in the directory
    regex_pattern : str
        Regex pattern with two capture groups for order and pixel, to extract the HEALPix pixels from the
        file names

    Returns
    -------
    list[HealpixPixel]
        List of HEALPix pixels extracted from the file names in the directory that match the glob pattern.
    """
    file_pattern = re.compile(regex_pattern)

    def get_matches(path: UPath):
        match = file_pattern.match(path.name)
        if match is None:
            raise ValueError(f"File {path} does not match the expected pattern {regex_pattern}")
        return match.group(1, 2)

    pixel_tuples = [get_matches(path) for path in (path.glob(glob_pattern))]
    return [HealpixPixel(int(match[0]), int(match[1])) for match in pixel_tuples]


def write_histogram(histogram: SparseHistogram, base_catalog_path: UPath, pixel: HealpixPixel):
    """Writes the sparse histogram for a partition to a file in the respective pixel directory.

    Parameters
    ----------
    histogram : SparseHistogram
        The sparse count histogram for the partition, at the specified order.
    base_catalog_path : path-like
        Location of the base catalog directory to write to
    pixel : HealpixPixel
        HEALPix pixel of file to be written
    """
    histogram_path = base_catalog_path / HISTOGRAM_DIR_NAME
    histogram_path.mkdir(exist_ok=True)
    hist_file = histogram_path / f"{pixel.order}_{pixel.pixel}_histogram.npz"
    histogram.to_file(hist_file)


def read_histograms(base_catalog_path: UPath):
    """Reads the sparse histograms for all partitions from the respective files in the histogram directory.

    Parameters
    ----------
    base_catalog_path : path-like
        Location of the base catalog directory to read from

    Returns
    -------
    tuple[list[HealpixPixel], list[SparseHistogram], list[int]]
        A tuple with the list of pixels for which histograms were read, the list of respective
        sparse histograms, and the list of total counts for each partition
        (i.e. the sum of the histogram counts).
    """
    histogram_path = base_catalog_path / HISTOGRAM_DIR_NAME
    pixels = _read_pixels_from_files(histogram_path, "*_histogram.npz", r"(\d+)_(\d+)_histogram.npz")
    histograms = [
        SparseHistogram.from_file(histogram_path / f"{pixel.order}_{pixel.pixel}_histogram.npz")
        for pixel in pixels
    ]
    lens = [sum(hist.counts) for hist in histograms]
    return pixels, histograms, lens


def remove_histogram_files(base_catalog_path: UPath):
    """Removes the histogram files from the histogram directory.

    Parameters
    ----------
    base_catalog_path : path-like
        Location of the base catalog directory to remove histogram files from
    """
    histogram_path = base_catalog_path / HISTOGRAM_DIR_NAME
    file_io.remove_directory(histogram_path, ignore_errors=True)


def write_done_pixel(base_catalog_path: UPath, pixel: HealpixPixel):
    """Writes an empty file in the respective pixel directory to indicate that the partition has been written.

    Parameters
    ----------
    base_catalog_path : path-like
        Location of the base catalog directory to write to
    pixel : HealpixPixel
        HEALPix pixel of file to be written
    """
    done_path = base_catalog_path / DONE_DIR_NAME
    done_path.mkdir(exist_ok=True)
    done_file = done_path / f"{pixel.order}_{pixel.pixel}_done"
    done_file.touch()


def read_done_pixels(base_catalog_path: UPath):
    """Reads the done files in the done directory to get the list of pixels that partitions have been written.

    Parameters
    ----------
    base_catalog_path : path-like
        Location of the base catalog directory to read from

    Returns
    -------
    list[HealpixPixel]
        List of HEALPix pixels for which done files were found in the done directory, indicating
        that the respective partitions have been written.
    """
    done_path = base_catalog_path / DONE_DIR_NAME
    return _read_pixels_from_files(done_path, "*_done", r"(\d+)_(\d+)_done")


def remove_done_files(base_catalog_path: UPath):
    """Removes the done files from the done directory.

    Parameters
    ----------
    base_catalog_path : path-like
        Location of the base catalog directory to remove done files from
    """
    done_path = base_catalog_path / DONE_DIR_NAME
    file_io.remove_directory(done_path, ignore_errors=True)


# pylint: disable=protected-access,too-many-locals
def to_hats(
    catalog: HealpixDataset,
    *,
    base_catalog_path: str | Path | UPath,
    catalog_name: str | None = None,
    default_columns: list[str] | None = None,
    histogram_order: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    progress_bar: bool = True,
    tqdm_kwargs=None,
    create_thumbnail: bool = False,
    skymap_alt_orders: list[int] | None = None,
    addl_hats_properties: dict | None = None,
    error_if_empty: bool = True,
    **kwargs,
):
    """Writes a catalog to disk, in HATS format.

    The output catalog comprises  partitioned parquet files and respective metadata,
    as well as text and CSV files detailing partition, catalog and provenance info.

    This will only write a SINGLE catalog, so if this catalog contains margins,
    you should use ``to_collection`` to write all parts of the catalog together.

    Parameters
    ----------
    catalog : HealpixDataset
        A catalog to export
    base_catalog_path : path-like
        Location where catalog is saved to
    catalog_name : str or None, default None
        The name of the output catalog
    default_columns : list[str] or None, default None
        A metadata property with the list of the columns in the
        catalog to be loaded by default. Uses the default columns from the original hats
        catalog if they exist.
    histogram_order : int or None, default None
        The default order for the count histogram. Defaults to the same
        skymap order as original catalog, or the highest order healpix of the current
        catalog data partitions.
    overwrite : bool, default False
        If True existing catalog is overwritten
    resume : bool, default False
        If True, will attempt to resume a previously interrupted write to the same directory.
    progress_bar : bool, default True
        If True, shows a progress bar for the export process. Defaults to True.
    tqdm_kwargs : dict, default None
        Additional kwargs to pass to the tqdm progress bar.
    create_thumbnail : bool, default False
        If True, create a data thumbnail of the catalog for
        previewing purposes. Defaults to False.
    skymap_alt_orders : list[int] or None, default None
        We will write a skymap file at the ``histogram_order``,
        but can also write down-sampled skymaps, for easier previewing of the data.
    addl_hats_properties : dict or None, default None
        key-value pairs of additional properties to write in the
        ``hats.properties`` file.
    error_if_empty : bool, default True
        If True, raises an error if the output catalog is empty
    **kwargs :
        Arguments to pass to the parquet write operations
    """
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be True")
    # Create the output directory for the catalog
    base_catalog_path = hc.io.file_io.get_upath(base_catalog_path)
    existing_pixels: list[HealpixPixel] = []
    histograms: list[SparseHistogram] = []
    counts: list[int] = []
    if hc.io.file_io.directory_has_contents(base_catalog_path):
        if not overwrite and not resume:
            raise ValueError(
                f"base_catalog_path ({str(base_catalog_path)}) contains files."
                " choose a different directory or set overwrite or resume to True."
            )
        if overwrite:
            hc.io.file_io.remove_directory(base_catalog_path)
        if resume:
            existing_pixels, histograms, counts = _read_existing_progress(base_catalog_path, catalog)
    hc.io.file_io.make_directory(base_catalog_path, exist_ok=True)
    if histogram_order is None:
        if catalog.hc_structure.catalog_info.skymap_order is not None:
            histogram_order = catalog.hc_structure.catalog_info.skymap_order
        else:
            max_catalog_depth = (
                catalog.hc_structure.pixel_tree.get_max_depth()
                if len(catalog.get_healpix_pixels()) > 0
                else 0
            )
            histogram_order = max(max_catalog_depth, 8)
    # Save partition parquet files
    kwargs = set_default_write_table_kwargs(kwargs)

    desc = tqdm_kwargs.pop("desc", "Writing Catalog") if tqdm_kwargs else "Writing Catalog"

    with TqdmCallback(desc=desc, disable=not progress_bar, **(tqdm_kwargs or {})):
        new_pixels, new_counts, new_histograms = write_partitions(
            catalog,
            base_catalog_dir_fp=base_catalog_path,
            histogram_order=histogram_order,
            existing_pixels=existing_pixels,
            **kwargs,
        )
        pixels = existing_pixels + new_pixels
        histograms = histograms + new_histograms
        counts = counts + new_counts

    # Check that the catalog is not empty
    if error_if_empty and len(pixels) == 0:
        raise RuntimeError("The output catalog is empty")

    # Save parquet metadata and create a data thumbnail if needed
    hats_max_rows = int(max(counts)) if counts else 0

    _write_parquet_metadata(base_catalog_path, catalog, create_thumbnail, hats_max_rows, pixels)
    # Save partition info
    PartitionInfo(pixels).write_to_file(base_catalog_path / "partition_info.csv")

    # Save catalog info
    if default_columns:
        _validate_default_columns(catalog, default_columns)
    else:
        default_columns = None

    if not addl_hats_properties:
        addl_hats_properties = {}

    if catalog.hc_structure.catalog_info.catalog_type in (CatalogType.OBJECT, CatalogType.SOURCE):
        addl_hats_properties = addl_hats_properties | {
            "skymap_order": int(histogram_order),
            "skymap_alt_orders": skymap_alt_orders,
        }

        _write_skymaps(base_catalog_path, histogram_order, histograms, skymap_alt_orders)

    addl_hats_properties = addl_hats_properties | new_provenance_properties(base_catalog_path)

    new_hc_structure = create_modified_catalog_structure(
        catalog.hc_structure,
        base_catalog_path,
        catalog_name if catalog_name else catalog.hc_structure.catalog_name,
        total_rows=int(np.sum(counts)),
        default_columns=default_columns,
        hats_max_rows=hats_max_rows,
        **addl_hats_properties,
    )
    new_hc_structure.catalog_info.to_properties_file(base_catalog_path)

    remove_histogram_files(base_catalog_path)
    remove_done_files(base_catalog_path)


def _read_existing_progress(
    base_catalog_path: UPath, catalog: HealpixDataset
) -> tuple[list[HealpixPixel], Any, Any]:
    """Read the done and histogram files from a previous write attempt to the same directory"""
    existing_pixels = read_done_pixels(base_catalog_path)
    hist_pixels, histograms, counts = read_histograms(base_catalog_path)
    for pixel in existing_pixels:
        if pixel not in catalog.hc_structure.pixel_tree:
            raise ValueError(
                f"Pixel {pixel} from existing catalog files is not present in the provided catalog."
                " Cannot resume write. Please check that the directory contains files from this "
                " catalog, or set resume to False and overwrite to True to overwrite to this"
                " directory instead of resuming."
            )
    if len(hist_pixels) != len(existing_pixels) or set(hist_pixels) != set(existing_pixels):
        raise ValueError(
            "The existing histogram files in the directory do not match the done pixels."
            " Cannot resume write. Please check that the directory contains files from this catalog,"
            " or set resume to False and overwrite to True to overwrite to this directory instead of"
            " resuming."
        )
    return existing_pixels, histograms, counts


def _write_parquet_metadata(
    base_catalog_path: UPath,
    catalog: HealpixDataset,
    create_thumbnail: bool,
    hats_max_rows: int,
    pixels: list[Any],
):
    """Writes the parquet metadata for the catalog. If there are no pixels,
    writes empty metadata files with the correct schema."""
    if len(pixels) > 0:
        hc.io.write_parquet_metadata(
            base_catalog_path, create_thumbnail=create_thumbnail, thumbnail_threshold=hats_max_rows
        )
    else:
        metadata_path = hc.io.paths.get_parquet_metadata_pointer(base_catalog_path)
        common_metadata_path = hc.io.paths.get_common_metadata_pointer(base_catalog_path)
        file_io.make_directory(metadata_path.parent, exist_ok=True)
        pq.write_metadata(catalog.hc_structure.schema, metadata_path.path, filesystem=metadata_path.fs)
        pq.write_metadata(
            catalog.hc_structure.schema, common_metadata_path.path, filesystem=common_metadata_path.fs
        )


def _write_skymaps(
    base_catalog_path: UPath,
    histogram_order: int,
    histograms: list[Any] | Any,
    skymap_alt_orders: list[int] | None,
):
    """Writes the skymap files for the catalog, based on the histograms for each partition."""
    # Save the point distribution map
    total_histogram = HistogramAggregator(histogram_order)
    for partition_hist in histograms:
        total_histogram.add(partition_hist)
    point_map_path = hc.io.paths.get_point_map_file_pointer(base_catalog_path)
    full_histogram = total_histogram.full_histogram
    hc.io.file_io.write_fits_image(full_histogram, point_map_path)

    write_skymap(histogram=full_histogram, catalog_dir=base_catalog_path, orders=skymap_alt_orders)


def _validate_default_columns(catalog: HealpixDataset, default_columns: list[str]):
    """Checks that the provided default columns are valid"""
    # Check if any of the default columns is missing
    missing_columns = set(default_columns) - set(catalog._ddf.exploded_columns)
    if missing_columns:
        raise ValueError(f"Default columns `{missing_columns}` not found in catalog")
    # Check for full and partial load of the same column and error
    all_subcolumns = catalog._ddf._meta.get_subcolumns()
    for col in default_columns:
        if col in all_subcolumns:
            nested_col = col.split(".")[0]
            if nested_col in default_columns:
                raise ValueError(
                    f"The provided default column list contains both a "
                    f"full and partial load of the column '{nested_col}'. "
                    f"Please either remove the partial load or the full load."
                )


def write_partitions(
    catalog: HealpixDataset,
    base_catalog_dir_fp: str | Path | UPath,
    histogram_order: int,
    existing_pixels: list[HealpixPixel] | None = None,
    **kwargs,
) -> tuple[list[HealpixPixel], list[int], list[SparseHistogram]]:
    """Saves catalog partitions as parquet to disk and computes the sparse
    count histogram for each partition. The histogram is either of order 8
    or the maximum pixel order in the catalog, whichever is greater.

    Parameters
    ----------
    catalog : HealpixDataset
        A catalog to export
    base_catalog_dir_fp : path-like
        Path to the base directory of the catalog
    histogram_order : int
        The order of the count histogram to generate
    **kwargs
        Arguments to pass to the parquet write operations

    Returns
    -------
    tuple[list[HealpixPixel], list[int], list[SparseHistogram]]
        A tuple with the array of non-empty pixels, the array with the total counts
        as well as the array with the sparse count histograms.
    """
    base_catalog_dir_fp = hc.io.file_io.get_upath(base_catalog_dir_fp)
    pixels = []
    existing_pixels_set = set(existing_pixels) if existing_pixels is not None else set()

    for pixel in catalog.get_healpix_pixels():
        if pixel in existing_pixels_set:
            continue
        pixels.append(pixel)

    write_cat = catalog.partitions[pixels]

    res_cat = write_cat.map_partitions(
        perform_write,
        base_catalog_dir_fp,
        histogram_order,
        meta=WRITE_RESULT_META,
        include_pixel=True,
        **kwargs,
    )

    if len(pixels) > 0:
        results = res_cat.compute()
        counts, histograms = results["count"].tolist(), results["histogram"].tolist()
    else:
        counts, histograms = (), ()

    non_empty_indices = np.nonzero(counts)
    non_empty_pixels = np.array(pixels)[non_empty_indices]
    non_empty_counts = np.array(counts)[non_empty_indices]
    non_empty_hists = np.array(histograms)[non_empty_indices]

    return list(non_empty_pixels), list(non_empty_counts), list(non_empty_hists)


def create_modified_catalog_structure(
    catalog_structure: HCHealpixDataset, catalog_base_dir: str | Path | UPath, catalog_name: str, **kwargs
) -> HCHealpixDataset:
    """Creates a modified version of the HATS catalog structure

    Parameters
    ----------
    catalog_structure : HCHealpixDataset
        HATS catalog structure
    catalog_base_dir : path-like
        Base location for the catalog
    catalog_name : str
        The name of the catalog to be saved
    **kwargs
        The remaining parameters to be updated in the catalog info object

    Returns
    -------
    HCHealpixDataset
        A HATS structure, modified with the parameters provided.
    """
    new_hc_structure = copy(catalog_structure)
    new_hc_structure.catalog_name = catalog_name
    new_hc_structure.catalog_path = catalog_base_dir
    new_hc_structure.catalog_base_dir = hc.io.file_io.get_upath(catalog_base_dir)

    new_hc_structure.catalog_info = new_hc_structure.catalog_info.copy_and_update(**kwargs)
    new_hc_structure.catalog_info.catalog_name = catalog_name
    return new_hc_structure
