import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from hats.pixel_math.spatial_index import compute_spatial_index

from lsdb.catalog.image_catalog import wcs_to_header_string
from lsdb.cutouts import CoverageMap, CutoutArray, match_partition
from lsdb.loaders.images.image_catalog_loader import image_footprint_moc

MOC_ORDER = 11
SIZE = 100


def make_wcs(ra, dec, pixscale_deg=0.005):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [SIZE / 2, SIZE / 2]
    wcs.wcs.cdelt = [-pixscale_deg, pixscale_deg]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def make_image_rows(specs):
    """specs: list of (image_id, ra, dec); returns image catalog-style rows."""
    rows = []
    for image_id, ra, dec in specs:
        wcs = make_wcs(ra, dec)
        ranges = np.asarray(image_footprint_moc(wcs, SIZE, SIZE, MOC_ORDER).to_depth29_ranges)
        rows.append(
            {
                "image_id": image_id,
                "path": f"s3://fake/{image_id}.fits",
                "width": SIZE,
                "height": SIZE,
                "wcs": wcs_to_header_string(wcs),
                "footprint_ranges": ranges.astype(np.int64).ravel().tolist(),
            }
        )
    return pd.DataFrame(rows)


def make_objects(coords):
    """coords: list of (ra, dec); returns a partition-like frame indexed by _healpix_29."""
    ra = [c[0] for c in coords]
    dec = [c[1] for c in coords]
    frame = pd.DataFrame({"ra": ra, "dec": dec})
    frame.index = pd.Index(compute_spatial_index(ra_values=ra, dec_values=dec), name="_healpix_29")
    return frame.sort_index()


# v1 and v2 overlap; the overlap contains v1's east edge and v2's west edge
IMAGE_SPECS = [("v1", 180.0, -30.0), ("v2", 180.3, -30.0)]


def test_coverage_map_structure():
    image_rows = make_image_rows(IMAGE_SPECS)
    coverage = CoverageMap.from_image_rows(image_rows)
    # Segments are sorted and disjoint
    assert np.all(coverage.ends > coverage.starts)
    assert np.all(coverage.starts[1:] >= coverage.ends[:-1])
    # Both single- and double-coverage segments exist (v1 and v2 overlap)
    assert set(np.unique(coverage.depth())) == {1, 2}


def test_coverage_map_lookup():
    image_rows = make_image_rows(IMAGE_SPECS)
    coverage = CoverageMap.from_image_rows(image_rows)
    v1_only = compute_spatial_index(ra_values=[179.9], dec_values=[-30.0])
    overlap = compute_spatial_index(ra_values=[180.15], dec_values=[-30.0])
    uncovered = compute_spatial_index(ra_values=[190.0], dec_values=[-40.0])
    segments = coverage.lookup_segments(np.concatenate([v1_only, overlap, uncovered]))
    assert segments[0] >= 0
    assert list(coverage.segment_images(segments[0])) == [0]
    assert list(coverage.segment_images(segments[1])) == [0, 1]
    assert segments[2] == -1


def test_match_basic():
    image_rows = make_image_rows(IMAGE_SPECS)
    objects = make_objects([(179.9, -30.0), (180.15, -30.0), (180.4, -30.0), (190.0, -40.0)])
    cutouts = match_partition(objects, image_rows, stamp_size=11, attach_store=False)
    assert isinstance(cutouts, CutoutArray)
    assert len(cutouts) == len(objects)
    descriptors = cutouts.descriptors()
    by_position = {round(objects["ra"].iloc[i], 2): i for i in range(len(objects))}
    assert descriptors["image_id"].iloc[by_position[179.9]] == "v1"
    assert descriptors["image_id"].iloc[by_position[180.4]] == "v2"
    assert descriptors["image_id"].iloc[by_position[180.15]] in ("v1", "v2")  # overlap: first fit
    assert cutouts[by_position[190.0]] is pd.NA


def test_match_descriptor_is_correct_projection():
    image_rows = make_image_rows(IMAGE_SPECS)
    objects = make_objects([(179.95, -30.02)])
    cutouts = match_partition(objects, image_rows, stamp_size=(9, 15), attach_store=False)
    ref = cutouts.descriptors().iloc[0]
    wcs = make_wcs(*IMAGE_SPECS[0][1:])
    x, y = wcs.wcs_world2pix([179.95], [-30.02], 0)
    assert ref["image_id"] == "v1"
    assert ref["x0"] == round(float(x[0])) - 15 // 2
    assert ref["y0"] == round(float(y[0])) - 9 // 2
    assert ref["width"] == 15
    assert ref["height"] == 9


def test_match_falls_through_to_second_image_at_edge():
    # An object in the overlap near v1's edge: a big stamp does not fit in v1
    # but fits in v2, so first-fit must fall through to v2
    image_rows = make_image_rows(IMAGE_SPECS)
    v1_wcs = make_wcs(*IMAGE_SPECS[0][1:])
    edge_coord = v1_wcs.pixel_to_world(2.0, 50.0)  # 2px from v1's edge, inside v2
    objects = make_objects([(edge_coord.ra.deg, edge_coord.dec.deg)])
    small = match_partition(objects, image_rows, stamp_size=3, attach_store=False)
    big = match_partition(objects, image_rows, stamp_size=21, attach_store=False)
    assert small.descriptors()["image_id"].iloc[0] == "v1"
    assert big.descriptors()["image_id"].iloc[0] == "v2"


def test_match_na_when_stamp_fits_nowhere():
    image_rows = make_image_rows([("v1", 180.0, -30.0)])
    v1_wcs = make_wcs(180.0, -30.0)
    edge_coord = v1_wcs.pixel_to_world(1.0, 1.0)
    objects = make_objects([(edge_coord.ra.deg, edge_coord.dec.deg)])
    assert match_partition(objects, image_rows, stamp_size=15, attach_store=False)[0] is pd.NA


def test_match_empty_inputs():
    image_rows = make_image_rows(IMAGE_SPECS)
    no_objects = match_partition(make_objects([]), image_rows, stamp_size=5, attach_store=False)
    assert len(no_objects) == 0
    objects = make_objects([(180.0, -30.0)])
    no_images = match_partition(objects, image_rows.iloc[:0], stamp_size=5, attach_store=False)
    assert len(no_images) == 1
    assert no_images[0] is pd.NA


def test_match_attaches_catalog_store():
    from lsdb.cutouts import CatalogImageStore

    image_rows = make_image_rows(IMAGE_SPECS)
    objects = make_objects([(179.9, -30.0)])
    cutouts = match_partition(objects, image_rows, stamp_size=5)
    assert isinstance(cutouts.store, CatalogImageStore)
    assert "v1" in cutouts.store


def test_match_result_builds_series():
    image_rows = make_image_rows(IMAGE_SPECS)
    objects = make_objects([(179.9, -30.0), (190.0, -40.0)])
    cutouts = match_partition(objects, image_rows, stamp_size=7, attach_store=False)
    series = pd.Series(cutouts, index=objects.index, name="cutouts")
    assert series.dtype == "cutout"
    assert series.notna().sum() == 1
