"""CoverageMap: a disjoint segmentation of the healpix29 line by image coverage.

Built from image footprint ranges (recomputed from the WCS parameters stored
in an image catalog) via a sweep line: the depth-29 HEALPix line is cut into
disjoint segments on which the set of covering images is constant. Object
lookup is then a single vectorized ``searchsorted`` against the segment
starts — the same sorted-interval machinery as HATS pixel tree alignment,
with image sets as payloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["CoverageMap"]


class CoverageMap:
    """Maps healpix29 positions to the images covering them.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries: segment ``i`` spans ``[starts[i], ends[i])``.
        Segments are sorted and disjoint; gaps between them are uncovered sky.
    offsets : np.ndarray
        Prefix offsets into ``image_positions``: segment ``i``'s images are
        ``image_positions[offsets[i]:offsets[i + 1]]``.
    image_positions : np.ndarray
        Flattened per-segment image lists, as positions into the image row
        set the map was built from.
    """

    def __init__(
        self, starts: np.ndarray, ends: np.ndarray, offsets: np.ndarray, image_positions: np.ndarray
    ):
        self.starts = starts
        self.ends = ends
        self.offsets = offsets
        self.image_positions = image_positions

    @classmethod
    def from_footprint_ranges(cls, footprint_ranges: list[np.ndarray]) -> CoverageMap:
        """Build a coverage map from per-image depth-29 ranges.

        Parameters
        ----------
        footprint_ranges : list of np.ndarray
            One ``(N, 2)`` array of sorted, disjoint ``[start, end)`` ranges
            per image. The list position of each image is the position
            reported in lookups.

        Returns
        -------
        CoverageMap
        """
        events = []  # (coordinate, +1 open / -1 close, image position)
        for position, ranges in enumerate(footprint_ranges):
            for start, end in ranges:
                events.append((int(start), 1, position))
                events.append((int(end), -1, position))
        events.sort()

        starts: list[int] = []
        ends: list[int] = []
        offsets: list[int] = [0]
        image_positions: list[int] = []
        active: set[int] = set()
        previous = None
        index = 0
        while index < len(events):
            coordinate = events[index][0]
            if previous is not None and active and coordinate > previous:
                starts.append(previous)
                ends.append(coordinate)
                image_positions.extend(sorted(active))
                offsets.append(len(image_positions))
            while index < len(events) and events[index][0] == coordinate:
                _, delta, position = events[index]
                if delta > 0:
                    active.add(position)
                else:
                    active.discard(position)
                index += 1
            previous = coordinate
        return cls(
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(image_positions, dtype=np.int64),
        )

    @classmethod
    def from_image_rows(cls, image_rows: pd.DataFrame, moc_order: int = 11) -> CoverageMap:
        """Build a coverage map from image catalog rows.

        Footprints are computed from each row's stored WCS parameters and
        dimensions; nothing footprint-related is read from disk.

        Parameters
        ----------
        image_rows : pd.DataFrame
            Image catalog rows with ``wcs``, ``width`` and ``height``
            columns. Image positions in lookups are positional indices into
            these rows.
        moc_order : int, default 11
            HEALPix order at which footprint MOCs are computed. Deeper
            orders hug the true footprints more tightly (fewer boundary
            false-positive candidates) at slightly higher build cost.

        Returns
        -------
        CoverageMap
        """
        # Imported here to avoid a circular import at module load time
        from lsdb.catalog.image_catalog import (  # pylint: disable=import-outside-toplevel
            image_footprint_moc,
            wcs_from_params,
        )

        footprint_ranges = [
            np.asarray(
                image_footprint_moc(
                    wcs_from_params(row["wcs"]), row["width"], row["height"], moc_order
                ).to_depth29_ranges,
                dtype=np.int64,
            )
            for _, row in image_rows.iterrows()
        ]
        return cls.from_footprint_ranges(footprint_ranges)

    def __len__(self) -> int:
        return len(self.starts)

    def lookup_segments(self, healpix29: np.ndarray) -> np.ndarray:
        """Find the covered segment of each healpix29 position.

        Parameters
        ----------
        healpix29 : np.ndarray
            Depth-29 HEALPix values (e.g. an object partition's
            ``_healpix_29`` index).

        Returns
        -------
        np.ndarray
            Segment index per position; -1 where the position is uncovered.
        """
        healpix29 = np.asarray(healpix29, dtype=np.int64)
        segment = np.searchsorted(self.starts, healpix29, side="right") - 1
        inside = (segment >= 0) & (healpix29 < self.ends[np.clip(segment, 0, None)])
        return np.where(inside, segment, -1)

    def segment_images(self, segment: int) -> np.ndarray:
        """The image positions covering one segment.

        Parameters
        ----------
        segment : int
            Segment index from :meth:`lookup_segments`.

        Returns
        -------
        np.ndarray
        """
        return self.image_positions[self.offsets[segment] : self.offsets[segment + 1]]

    def depth(self) -> np.ndarray:
        """The number of covering images per segment (the coverage depth map).

        Returns
        -------
        np.ndarray
        """
        return np.diff(self.offsets)
