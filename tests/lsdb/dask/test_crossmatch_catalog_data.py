from __future__ import annotations

import pandas as pd
import pytest
from hats.pixel_math import HealpixPixel

import lsdb
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.dask.crossmatch_catalog_data import perform_crossmatch
from lsdb.dask.merge_catalog_functions import filter_by_spatial_index_to_pixel


class RecordingCrossmatchAlgorithm(AbstractCrossmatchAlgorithm):
    extra_columns = None

    def __init__(self, seen):
        self.seen = seen

    def validate(self, left, right):
        # Skip validation for the lightweight test inputs
        return

    def crossmatch(self, crossmatch_args, how, suffixes, suffix_method="all_columns"):
        filtered_left = crossmatch_args.left_df.copy()
        self.seen.append(filtered_left)
        return filtered_left


@pytest.mark.filterwarnings("ignore:.*default suffix behavior.*")
def test_perform_crossmatch_filters_left_to_aligned_pixel(small_sky_order1_catalog, test_data_dir):
    left_pix = HealpixPixel(1, 44)
    right_pix = HealpixPixel(3, 707)

    left_catalog = small_sky_order1_catalog.pixel_search([left_pix])
    right_catalog = lsdb.open_catalog(
        test_data_dir / "small_sky_order3_source", margin_cache=test_data_dir / "small_sky_order3_source_margin"
    ).pixel_search([right_pix])

    left_df = left_catalog.get_partition(left_pix.order, left_pix.pixel).compute()
    right_df = right_catalog.get_partition(right_pix.order, right_pix.pixel).compute()
    meta_df = left_df.head(0)

    left_catalog_info = left_catalog.hc_structure.catalog_info
    right_catalog_info = right_catalog.hc_structure.catalog_info
    right_margin_info = right_catalog.margin.hc_structure.catalog_info

    aligned_pixels = [HealpixPixel(3, pixel) for pixel in range(704, 720)]

    seen_left = []
    algorithm = RecordingCrossmatchAlgorithm(seen_left)

    results = []
    for aligned_pixel in aligned_pixels:
        expected_filtered = filter_by_spatial_index_to_pixel(
            left_df,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=left_catalog_info.healpix_order,
        )

        previous_seen = len(seen_left)
        result = perform_crossmatch(
            left_df,
            right_df,
            None,
            None,
            left_pix,
            right_pix,
            right_pix,
            aligned_pixel,
            left_catalog_info,
            right_catalog_info,
            right_margin_info,
            None,
            algorithm,
            "left",
            ("_left", "_right"),
            "all_columns",
            meta_df,
        )

        results.append(result)

        assert len(seen_left) == previous_seen + 1
        pd.testing.assert_frame_equal(seen_left[-1], expected_filtered)

    combined = pd.concat([df for df in results if not df.empty]).sort_index()
    pd.testing.assert_index_equal(combined.index.sort_values(), left_df.index.sort_values())
    assert combined.index.duplicated().sum() == left_df.index.duplicated().sum()
