

import inspect
from typing import Callable, Iterable, Literal, Type

import nested_pandas as npd

from lsdb.catalog import Catalog
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.loaders.dataframe.from_dataframe import from_dataframe


def crossmatch(
        left: Catalog | npd.NestedFrame,
        right: Catalog | npd.NestedFrame,
        suffixes: tuple[str, str] | None = None,
        algorithm: (
            Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
        ) = BuiltInCrossmatchAlgorithm.KD_TREE,
        output_catalog_name: str | None = None,
        require_right_margin: bool = False,
        **kwargs,
    ) -> Catalog:
    """Crossmatch two nested pandas dataframes."""
    # # Lazy import to avoid circular dependencies.
    # # pylint: disable=C0415
    # from lsdb.loaders.dataframe.from_dataframe import from_dataframe

    # Separate kwargs for from_dataframe and crossmatch
    sig = inspect.signature(from_dataframe)
    from_dataframe_arg_names = list(sig.parameters.keys())
    from_dataframe_kwargs = {k: kwargs.pop(k) for k in from_dataframe_arg_names if k in kwargs}


    # Check if the left dataframe is a NestedFrame, and if so, convert it to a Catalog.
    if not isinstance(left, Catalog):
        if not isinstance(left, npd.NestedFrame):
            raise TypeError(
                f"Left argument must be a NestedFrame or Catalog, not {type(left)}."
            )
        # Convert the left DataFrame to a Catalog.
        left = from_dataframe(left, **from_dataframe_kwargs)

    # Check if the right dataframe is a NestedFrame, and if so, convert it to a Catalog.
    if not isinstance(right, Catalog):
        if not isinstance(right, npd.NestedFrame):
            raise TypeError(
                f"Right argument must be a NestedFrame or Catalog, not {type(right)}."
            )
        # Convert the right DataFrame to a Catalog.
        right = from_dataframe(right, **from_dataframe_kwargs)

    # Call the crossmatch method with the newly generated Catalog.
    return Catalog.crossmatch(
        left, right, suffixes, algorithm, output_catalog_name, require_right_margin, **kwargs
    )