import hats as hc
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath


class PathGenerator:
    """Generates custom paths for leaf HEALPix parquet files/directories."""

    def __init__(self, hc_catalog: HCHealpixDataset, query_url_params: dict | None = None):
        self.catalog_base_dir = hc.io.file_io.get_upath(hc_catalog.catalog_base_dir)
        self.npix_suffix = hc_catalog.catalog_info.npix_suffix
        self.query_url_params = query_url_params

    def __call__(self, pixel: HealpixPixel) -> UPath:
        return hc.io.pixel_catalog_file(
            catalog_base_dir=self.catalog_base_dir,
            pixel=pixel,
            query_params=self.query_url_params,
            npix_suffix=self.npix_suffix,
        )
