import hats as hc
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath


class PathGenerator:
    """Generates custom paths for loading parquet Healpix pixel files."""

    def __init__(self, hc_catalog, query_url_params=None):
        self.base_dir = hc_catalog.catalog_base_dir
        self.npix_suffix = hc_catalog.catalog_info.npix_suffix
        self.query_url_params = query_url_params

    def __call__(self, pixel: HealpixPixel) -> UPath:
        return hc.io.pixel_catalog_file(
            catalog_base_dir=self.base_dir,
            pixel=pixel,
            query_params=self.query_url_params,
            npix_suffix=self.npix_suffix,
        )
