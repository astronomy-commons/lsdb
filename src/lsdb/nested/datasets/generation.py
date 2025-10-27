import numpy as np
from astropy.coordinates import Angle, SkyCoord
from cdshealpix.nested import healpix_to_lonlat
from nested_pandas import datasets

import lsdb
from lsdb.core.search.region_search import BoxSearch, ConeSearch, PixelSearch
from lsdb.nested.core import NestedFrame


def generate_data(
    n_base, n_layer, npartitions=1, seed=None, ra_range=(0.0, 360.0), dec_range=(-90, 90), search_region=None
):
    """Generates a toy dataset.

    Docstring copied from nested-pandas.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    npartitions: int
        The number of partitions to split the data into.
    seed : int
        A seed to use for random generation of data
    ra_range : tuple
        A tuple of the min and max values for the ra column in degrees
    dec_range : tuple
        A tuple of the min and max values for the dec column in degrees
    search_region : AbstractSearch
        A search region to apply to the generated data. Currently supports the
        ConeSearch, BoxSearch, and PixelSearch regions. Note that if provided,
        this will override the `ra_range` and `dec_range` parameters.

    Returns
    -------
    NestedFrame
        The constructed Dask NestedFrame.

    Examples
    --------
    >>> from lsdb.nested.datasets import generate_data
    >>> nf = generate_data(10,100)
    >>> nf = generate_data(10, {"nested_a": 100, "nested_b": 200})

    Constraining spatial ranges:
    >>> nf = generate_data(10, 100, ra_range=(0., 10.), dec_range=(-5., 0.))

    Using a search region:
    >>> from lsdb import ConeSearch
    >>> nf = generate_data(10, 100, search_region=ConeSearch(5, 5, 1))
    """

    # Use nested-pandas generator
    base_nf = datasets.generate_data(n_base, n_layer, seed=seed)

    # Generated "ra" and "dec" columns for hats catalog validity
    rng = np.random.default_rng(seed)  # Use the provided seed for reproducibility

    if search_region is not None:
        if isinstance(search_region, ConeSearch):
            base_nf["ra"], base_nf["dec"] = _generate_cone_search(search_region, n_base=n_base, seed=seed)
        elif isinstance(search_region, BoxSearch):
            ra_range = search_region.ra
            dec_range = search_region.dec
            base_nf["ra"], base_nf["dec"] = _generate_box_radec(ra_range, dec_range, n_base, seed=seed)
        elif isinstance(search_region, PixelSearch):
            base_nf["ra"], base_nf["dec"] = _generate_pixel_search(search_region, n_base, seed=seed)
        else:
            raise NotImplementedError(
                "Only ConeSearch, BoxSearch and PixelSearch are currently supported for search_region"
            )

    else:
        # Generate random RA and Dec values within the specified ranges
        base_nf["ra"], base_nf["dec"] = _generate_box_radec(ra_range, dec_range, n_base, seed=seed)

    # Generate a random integer ID column, unique for each row
    base_nf["id"] = rng.choice(range(1, n_base * 10), size=n_base, replace=False)

    # Add an error column to the nested column
    if isinstance(n_layer, dict):
        for layer in n_layer.keys():
            base_nf[f"{layer}.flux_err"] = base_nf[f"{layer}.flux"] * 0.05
    else:
        base_nf["nested.flux_err"] = base_nf["nested.flux"] * 0.05

    # reorder columns nicely
    base_nf = base_nf[["ra", "dec", "id", "a", "b"] + base_nf.nested_columns]

    # Convert to lsdb.nested NestedFrame
    base_nf = NestedFrame.from_pandas(base_nf).repartition(npartitions=npartitions)

    return base_nf


def _generate_cone_search(
    cone_search: ConeSearch, n_base: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    ra_center = cone_search.ra
    dec_center = cone_search.dec

    # Generate RA/decs from the cone parameters
    radius_degrees = cone_search.radius_arcsec / 3600.0
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_base)
    cos_radius = np.cos(np.radians(radius_degrees))
    theta = np.arccos(rng.uniform(cos_radius, 1.0, size=n_base))
    cone_center = SkyCoord(ra=ra_center, dec=dec_center, unit="deg")
    coords = cone_center.directional_offset_by(
        position_angle=Angle(phi, "radian"), separation=Angle(theta, "radian")
    )
    return coords.ra.deg, coords.dec.deg


def _generate_box_radec(ra_range, dec_range, n_base, seed=None):
    """Generates a random set of RA and Dec values within a given range.

    Parameters
    ----------
    ra_range : tuple
        A tuple of the min and max values for the ra column in degrees
    dec_range : tuple
        A tuple of the min and max values for the dec column in degrees
    n_base : int
        The number of rows to generate for the base layer
    seed : int
        A seed to use for random generation of data

    Returns
    -------
    np.ndarray
        An array of shape (n_base, 2) containing the generated RA and Dec values.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(ra_range[0], ra_range[1], size=n_base)
    sindec_min = np.sin(np.radians(dec_range[0]))
    sindec_max = np.sin(np.radians(dec_range[1]))
    dec_rad = np.arcsin(rng.uniform(sindec_min, sindec_max, size=n_base))
    dec = np.degrees(dec_rad)
    return ra, dec


def _generate_pixel_search(
    pixel_search: PixelSearch, n_base: int, seed=None, gen_order: int = 29
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    pix_orders, pix_indexes = np.array([[pix.order, pix.pixel] for pix in pixel_search.pixels]).T
    if np.any(pix_orders > gen_order):
        raise ValueError("Pixel orders cannot be greater than gen_order")

    # Distribute points over pixels, inverse proportional to pixel areas
    inverse_area = 1.0 / (1 << (2 * pix_orders))
    prob = inverse_area / np.sum(inverse_area)
    pixels = rng.choice(len(pix_orders), p=prob, size=n_base)
    # Set-up index ranges for all points
    low_gen_index = pix_indexes[pixels] << (2 * (gen_order - pix_orders[pixels]))
    high_gen_index = (pix_indexes[pixels] + 1) << (2 * (gen_order - pix_orders[pixels]))
    # Generate healpix index on a high order
    gen_index = rng.integers(low_gen_index, high_gen_index)
    # Convert to ra and dec
    lon, lat = healpix_to_lonlat(gen_index, gen_order)
    return np.asarray(lon.deg), np.asarray(lat.deg)


def generate_catalog(
    n_base, n_layer, seed=None, ra_range=(0.0, 360.0), dec_range=(-90, 90), search_region=None, **kwargs
):
    """Generates a toy catalog.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    seed : int
        A seed to use for random generation of data
    ra_range : tuple
        A tuple of the min and max values for the ra column in degrees
    dec_range : tuple
        A tuple of the min and max values for the dec column in degrees
    search_region : AbstractSearch
        A search region to apply to the generated data. Currently supports the
        ConeSearch and BoxSearch regions. Note that if provided,
        this will override the `ra_range` and `dec_range` parameters.
    **kwargs :
        Additional keyword arguments to pass to `lsdb.from_dataframe`.

    Returns
    -------
    Catalog
        The constructed LSDB Catalog.

    Examples
    --------
    >>> from lsdb.nested.datasets import generate_catalog
    >>> gen_cat = generate_catalog(10,100)
    >>> gen_cat = generate_catalog(1000, 10, ra_range=(0.,10.), dec_range=(-5.,0.))

    Constraining spatial ranges:

    >>> gen_cat = generate_data(10, 100, ra_range=(0., 10.), dec_range=(-5., 0.))

    Using a search region:

    >>> from lsdb import ConeSearch
    >>> gen_cat = generate_data(10, 100, search_region=ConeSearch(5, 5, 1))
    """

    base_nf = generate_data(
        n_base, n_layer, seed=seed, ra_range=ra_range, dec_range=dec_range, search_region=search_region
    )
    # Give it a default catalog name if not provided
    if "catalog_name" not in kwargs:
        return lsdb.from_dataframe(base_nf.compute(), catalog_name="generated_catalog", **kwargs)
    return lsdb.from_dataframe(base_nf.compute(), **kwargs)
