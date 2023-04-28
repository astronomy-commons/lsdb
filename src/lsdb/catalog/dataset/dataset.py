import hipscat as hc
import dask.dataframe as dd


class Dataset:

    def __init__(
        self,
        ddf: dd.core.DataFrame,
        hc_structure: hc.catalog.Dataset,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
        """
        self._ddf = ddf
        self.hc_structure = hc_structure

    def __repr__(self):
        return self._ddf.__repr__()

    def _repr_html_(self):
        return self._ddf._repr_html_()

    def compute(self):
        """Compute dask distributed dataframe to pandas dataframe"""
        return self._ddf.compute()
