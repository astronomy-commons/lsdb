"""Per-partition matching of objects to images, producing cutout columns.

The flow, per partition:

1. Build a :class:`CoverageMap` from the partition's image rows (footprints
   recomputed from their stored WCS parameters) — a disjoint healpix29
   segmentation with image sets.
2. One vectorized lookup maps every object's ``_healpix_29`` to its segment,
   and through it to its candidate images. Objects on uncovered sky resolve
   to NA immediately.
3. First-fit selection: for rank r = 0, 1, ..., project each unresolved
   object into its rank-r candidate image (grouped by image, one vectorized
   ``world_to_pixel`` call per image) and accept the image if the full stamp
   fits inside it. This same projection filters out MOC boundary
   false-positives, so range over-coverage never produces a bad descriptor.

The result is a :class:`CutoutArray` aligned with the objects, holding NA
where no candidate image can host the stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from lsdb.cutouts.catalog_image_store import CatalogImageStore
from lsdb.cutouts.coverage_map import CoverageMap
from nested_pandas import CutoutArray
from nested_pandas.tensor.cutouts import CUTOUT_DESCRIPTOR_TYPE as CUTOUT_ARROW_TYPE
from lsdb.cutouts.image_store import ImageStore

__all__ = ["match_partition"]


def _resolve_stamp_size(stamp_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(stamp_size, int):
        return stamp_size, stamp_size
    height, width = stamp_size
    return int(height), int(width)


def match_partition(  # pylint: disable=too-many-locals
    objects: pd.DataFrame,
    image_rows: pd.DataFrame,
    stamp_size: int | tuple[int, int],
    ra_column: str = "ra",
    dec_column: str = "dec",
    moc_order: int = 11,
    store: ImageStore | None = None,
    attach_store: bool = True,
) -> CutoutArray:
    """Match one object partition against its overlapping images.

    Parameters
    ----------
    objects : pd.DataFrame
        Object partition, indexed by ``_healpix_29`` (the standard HATS
        spatial index) and sorted by it.
    image_rows : pd.DataFrame
        The image catalog rows overlapping this partition, including the
        ``wcs``, ``width``, ``height``, ``image_id`` and ``path`` columns.
    stamp_size : int or (int, int)
        Cutout size in pixels, as a square side or ``(height, width)``. The
        selected image must contain the full stamp around the object.
    ra_column : str, default "ra"
        Name of the object right ascension column.
    dec_column : str, default "dec"
        Name of the object declination column.
    moc_order : int, default 11
        HEALPix order at which image footprints are computed for candidate
        generation (typically the catalog's ``image_moc_order`` property).
    store : ImageStore, optional
        Image store to attach to the resulting cutout column. If None and
        ``attach_store`` is True, a :class:`CatalogImageStore` over
        ``image_rows`` is created.
    attach_store : bool, default True
        Whether to attach a store to the result (descriptors-only if False).

    Returns
    -------
    CutoutArray
        One cutout descriptor per object row (NA where no image fits),
        aligned with ``objects``.
    """
    from lsdb.catalog.image_catalog import (  # pylint: disable=import-outside-toplevel
        image_footprint_moc,
        wcs_from_params,
    )

    height, width = _resolve_stamp_size(stamp_size)
    n_objects = len(objects)

    # Descriptor fields being filled in; -1 image position means unresolved
    chosen_image = np.full(n_objects, -1, dtype=np.int64)
    chosen_x0 = np.zeros(n_objects, dtype=np.int64)
    chosen_y0 = np.zeros(n_objects, dtype=np.int64)

    if len(image_rows) > 0 and n_objects > 0:
        image_rows = image_rows.drop_duplicates(subset="image_id").reset_index(drop=True)
        image_wcs = [wcs_from_params(value) for value in image_rows["wcs"]]
        image_width = image_rows["width"].to_numpy(dtype=np.int64)
        image_height = image_rows["height"].to_numpy(dtype=np.int64)
        footprint_ranges = [
            np.asarray(image_footprint_moc(wcs, width, height, moc_order).to_depth29_ranges, dtype=np.int64)
            for wcs, width, height in zip(image_wcs, image_width, image_height)
        ]
        coverage = CoverageMap.from_footprint_ranges(footprint_ranges)
        segments = coverage.lookup_segments(objects.index.to_numpy())

        ra = objects[ra_column].to_numpy(dtype=np.float64)
        dec = objects[dec_column].to_numpy(dtype=np.float64)

        covered = np.flatnonzero(segments >= 0)
        candidate_lists = [coverage.segment_images(segment) for segment in segments[covered]]
        unresolved = covered
        rank = 0
        max_rank = max((len(candidates) for candidates in candidate_lists), default=0)
        candidate_by_object = dict(zip(covered, candidate_lists))
        while len(unresolved) > 0 and rank < max_rank:
            # rank-r candidate of each still-unresolved object (or -1 if exhausted)
            candidates = np.array(
                [
                    candidate_by_object[obj][rank] if rank < len(candidate_by_object[obj]) else -1
                    for obj in unresolved
                ],
                dtype=np.int64,
            )
            still_unresolved = []
            for image_position in np.unique(candidates[candidates >= 0]):
                members = unresolved[candidates == image_position]
                x, y = image_wcs[image_position].wcs_world2pix(ra[members], dec[members], 0)
                x0 = np.round(x).astype(np.int64) - width // 2
                y0 = np.round(y).astype(np.int64) - height // 2
                fits_inside = (
                    (x0 >= 0)
                    & (y0 >= 0)
                    & (x0 + width <= image_width[image_position])
                    & (y0 + height <= image_height[image_position])
                )
                accepted = members[fits_inside]
                chosen_image[accepted] = image_position
                chosen_x0[accepted] = x0[fits_inside]
                chosen_y0[accepted] = y0[fits_inside]
                still_unresolved.extend(members[~fits_inside])
            # Objects whose candidate list is exhausted (rank-r candidate was -1)
            # are permanently unresolved and drop out here.
            unresolved = np.asarray(sorted(still_unresolved), dtype=np.int64)
            rank += 1

    resolved = chosen_image >= 0
    image_ids = np.full(n_objects, None, dtype=object)
    if resolved.any():
        image_ids[resolved] = image_rows["image_id"].to_numpy(dtype=object)[chosen_image[resolved]]
    struct = pa.StructArray.from_arrays(
        [
            pa.array(image_ids, type=pa.string()),
            pa.array(np.where(resolved, chosen_x0, 0), type=pa.int32()),
            pa.array(np.where(resolved, chosen_y0, 0), type=pa.int32()),
            pa.array(np.full(n_objects, width, dtype=np.int32)),
            pa.array(np.full(n_objects, height, dtype=np.int32)),
        ],
        names=["image_id", "x0", "y0", "width", "height"],
        mask=pa.array(~resolved),
    ).cast(CUTOUT_ARROW_TYPE)

    result_store = store
    if result_store is None and attach_store and len(image_rows) > 0:
        result_store = CatalogImageStore(image_rows)
    return CutoutArray(struct, store=result_store if attach_store else None)
