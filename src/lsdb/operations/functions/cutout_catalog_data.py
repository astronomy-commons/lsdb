# pylint: disable=duplicate-code
from __future__ import annotations

from typing import TYPE_CHECKING

import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment, PixelAlignmentType
from hats.pixel_tree.pixel_alignment import align_with_mocs

from lsdb.operations.functions.merge_catalog_functions import (
    align_and_apply,
    filter_by_spatial_index_to_pixel,
    get_aligned_pixels_from_alignment,
    get_healpix_pixels_from_alignment,
)
from lsdb.operations.operation import Operation

if TYPE_CHECKING:
    from lsdb.catalog import Catalog
    from lsdb.catalog.image_catalog import ImageCatalog


# pylint: disable=too-many-arguments, unused-argument
def perform_add_cutouts(
    catalog_partition: npd.NestedFrame,
    image_partition: npd.NestedFrame,
    _aligned_placeholder,
    catalog_pixel: HealpixPixel,
    image_pixel: HealpixPixel | None,
    aligned_pixel: HealpixPixel,
    catalog_structure: TableProperties,
    image_structure: TableProperties | None,
    _aligned_structure,
    column_name: str,
    stamp_size: int | tuple[int, int],
    moc_order: int,
    attach_store: bool,
):
    """Match one object partition against its image rows and append the cutout column.

    Parameters
    ----------
    catalog_partition : npd.NestedFrame
        Partition of the point-source catalog.
    image_partition : npd.NestedFrame
        The image catalog partition overlapping it (empty when the object
        partition lies outside the image catalog's coverage).
    catalog_pixel : HealpixPixel
        The HEALPix pixel of the catalog partition.
    image_pixel : HealpixPixel or None
        The HEALPix pixel of the image partition, or None when there is no
        overlapping image partition.
    aligned_pixel : HealpixPixel
        The output pixel of this task; the object partition is filtered to it
        when the alignment splits object partitions into finer pieces.
    catalog_structure : hats.catalog.TableProperties
        The catalog info of the object catalog.
    image_structure : hats.catalog.TableProperties or None
        The catalog info of the image catalog.
    column_name : str
        Name of the cutout column to append.
    stamp_size : int or (int, int)
        Cutout size in pixels.
    moc_order : int
        HEALPix order for match-time image footprints.
    attach_store : bool
        Whether to attach a CatalogImageStore to the cutout column.

    Returns
    -------
    npd.NestedFrame
        The catalog partition with the cutout column appended.
    """
    # Imported here so the operations layer has no import-time dependency on cutouts
    from lsdb.cutouts import match_partition  # pylint: disable=import-outside-toplevel

    if aligned_pixel.order > catalog_pixel.order:
        catalog_partition = filter_by_spatial_index_to_pixel(
            catalog_partition,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=catalog_structure.healpix_order,
        )
    catalog_partition = catalog_partition.sort_index()
    cutouts = match_partition(
        catalog_partition,
        image_partition,
        stamp_size,
        ra_column=catalog_structure.ra_column or "ra",
        dec_column=catalog_structure.dec_column or "dec",
        moc_order=moc_order,
        attach_store=attach_store,
    )
    result = catalog_partition.copy()
    result[column_name] = pd.Series(cutouts, index=result.index)
    return result


# pylint: disable=protected-access
def add_cutouts_catalog_data(
    point_catalog: Catalog,
    images: ImageCatalog,
    stamp_size: int | tuple[int, int],
    column_name: str,
    moc_order: int,
    attach_store: bool,
) -> tuple[Operation, PixelAlignment]:
    """Align an object catalog with an image catalog and append a lazy cutout column.

    Uses a LEFT alignment: every object partition is kept, including those
    outside the image catalog's coverage (their cutouts are NA). Where the
    image tree is finer than the object tree, object partitions are split to
    the aligned pixels.

    Parameters
    ----------
    point_catalog : Catalog
        The object catalog.
    images : ImageCatalog
        The image metadata catalog.
    stamp_size : int or (int, int)
        Cutout size in pixels.
    column_name : str
        Name of the cutout column to append.
    moc_order : int
        HEALPix order for match-time image footprints.
    attach_store : bool
        Whether cutout columns carry a CatalogImageStore for lazy rendering.

    Returns
    -------
    tuple[Operation, PixelAlignment]
        The LSDB Operation for the result and the pixel alignment used.
    """
    # Imported here so the operations layer has no import-time dependency on cutouts
    from lsdb.cutouts.cutout_array import CutoutArray  # pylint: disable=import-outside-toplevel

    meta = point_catalog.meta.copy()
    meta[column_name] = pd.Series(CutoutArray._from_sequence([]), index=meta.index)

    alignment = align_with_mocs(
        point_catalog.hc_structure.pixel_tree,
        images.hc_structure.pixel_tree,
        point_catalog.hc_structure.moc,
        images.hc_structure.moc,
        alignment_type=PixelAlignmentType.LEFT,
    )

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # The third, catalog-less mapping delivers the aligned pixel to each task so
    # object partitions can be filtered to their aligned piece (LEFT alignments can
    # split a partition into pieces with and without an image partition).
    op = align_and_apply(
        [(point_catalog, left_pixels), (images, right_pixels), (None, aligned_pixels)],
        perform_add_cutouts,
        meta,
        aligned_pixels,
        column_name,
        stamp_size,
        moc_order,
        attach_store,
    )
    return op, alignment
