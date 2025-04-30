from nested_pandas import datasets
import numpy as np

from lsdb.nested.core import NestedFrame
import lsdb


def generate_data(n_base, n_layer, npartitions=1, 
                  seed=None, ra_range=(0.0,360.0), dec_range=(-90,90)) -> NestedFrame:
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

    Returns
    -------
    NestedFrame
        The constructed Dask NestedFrame.

    Examples
    --------
    >>> from lsdb.nested.datasets import generate_data # doctest: +SKIP
    >>> generate_data(10,100) # doctest: +SKIP
    >>> generate_data(10, {"nested_a": 100, "nested_b": 200}) # doctest: +SKIP
    """

    # Use nested-pandas generator
    base_nf = datasets.generate_data(n_base, n_layer, seed=seed)

    # Generated "ra" and "dec" columns for hats catalog validity
    rng = np.random.default_rng(seed)  # Use the provided seed for reproducibility
    base_nf["ra"] = rng.uniform(ra_range[0], ra_range[1], size=n_base)
    base_nf["dec"] = rng.uniform(dec_range[0], dec_range[1], size=n_base)

    # Convert to lsdb.nested NestedFrame
    base_nf = NestedFrame.from_pandas(base_nf).repartition(npartitions=npartitions)

    return base_nf

def generate_catalog(n_base, n_layer,
                     seed=None, ra_range=(0.0,360.0), dec_range=(-90,90), **kwargs):
    """Generates a toy catalog.

    Docstring copied from nested-pandas.

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
    **kwargs :
        Additional keyword arguments to pass to `lsdb.from_dataframe`.

    Returns
    -------
    Catalog
        The constructed LSDB Catalog.

    Examples
    --------
    >>> from lsdb.nested.datasets import generate_catalog # doctest: +SKIP
    >>> generate_catalog(10,100) # doctest: +SKIP
    >>> generate_catalog(1000, 10, ra_range=(0.,10.), dec_range=(-5.,0.)) # doctest: +SKIP
    """

    base_nf = generate_data(n_base, n_layer, seed=seed, ra_range=ra_range, dec_range=dec_range)
    return lsdb.from_dataframe(base_nf.compute(), **kwargs)

