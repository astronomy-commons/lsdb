import hats as hc
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath


class PathGenerator:
    """Generates custom paths for leaf HEALPix parquet files/directories."""

    catalog_base_dir: UPath | None = None
    """Path to the catalog base directory"""

    npix_suffix: str = ".parquet"
    """Suffix to the pixel partition"""

    query_url_params: dict | None = None
    """Dictionary of URL parameters with `columns` and `filters` attributes"""

    def set_internal_info(
        self, catalog_base_dir: UPath | None, npix_suffix: str, query_url_params: dict | None = None
    ):
        """Set catalog information and URL params"""
        self.catalog_base_dir = catalog_base_dir
        self.npix_suffix = npix_suffix
        self.query_url_params = query_url_params

    def __call__(self, pixel: HealpixPixel) -> UPath:
        return hc.io.pixel_catalog_file(
            catalog_base_dir=self.catalog_base_dir,
            pixel=pixel,
            query_params=self.query_url_params,
            npix_suffix=self.npix_suffix,
        )
