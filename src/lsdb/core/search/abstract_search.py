from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

import astropy
import nested_pandas as npd
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import BaseFrame
from hats.catalog import TableProperties
from hats.inspection.visualize_catalog import initialize_wcs_axes
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


# pylint: disable=too-many-instance-attributes, too-many-arguments
class AbstractSearch(ABC):
    """Abstract class used to write a reusable search query.

    These consist of two parts:
        - partition search - a (usually) coarse method of restricting
          the search space to just the partitions(/pixels) of interest
        - point search - a (usally) finer grained method to find
          individual rows matching the query terms.
    """

    def __init__(self, fine: bool = True):
        self.fine = fine

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        """Determine the target partitions for further filtering."""
        raise NotImplementedError("Search Class must implement `filter_hc_catalog` method")

    @abstractmethod
    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""

    def plot(
        self,
        projection: str = "MOL",
        title: str = "",
        fov: Quantity | tuple[Quantity, Quantity] | None = None,
        center: SkyCoord | None = None,
        wcs: astropy.wcs.WCS | None = None,
        frame_class: Type[BaseFrame] | None = None,
        ax: WCSAxes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        """Plot the search region

        Args:
            projection (str): The projection to use in the WCS. Available projections listed at
                https://docs.astropy.org/en/stable/wcs/supported_projections.html
            title (str): The title of the plot
            fov (Quantity or Sequence[Quantity, Quantity] | None): The Field of View of the WCS. Must be an
                astropy Quantity with an angular unit, or a tuple of quantities for different longitude and
                latitude FOVs (Default covers the full sky)
            center (SkyCoord | None): The center of the projection in the WCS (Default: SkyCoord(0, 0))
            wcs (WCS | None): The WCS to specify the projection of the plot. If used, all other WCS parameters
                are ignored and the parameters from the WCS object is used.
            frame_class (Type[BaseFrame] | None): The class of the frame for the WCSAxes to be initialized
                with. if the `ax` kwarg is used, this value is ignored (By Default uses EllipticalFrame for
                full sky projection. If FOV is set, RectangularFrame is used)
            ax (WCSAxes | None): The matplotlib axes to plot onto. If None, an axes will be created to be
                used. If specified, the axes must be an astropy WCSAxes, and the `wcs` parameter must be set
                with the WCS object used in the axes. (Default: None)
            fig (Figure | None): The matplotlib figure to add the axes to. If None, one will be created,
                unless ax is specified (Default: None)
            **kwargs: Additional kwargs to pass to creating the matplotlib patch object for the search region

        Returns:
            Tuple[Figure, WCSAxes] - The figure and axes used for the plot
        """
        fig, ax, wcs = initialize_wcs_axes(
            projection=projection,
            fov=fov,
            center=center,
            wcs=wcs,
            frame_class=frame_class,
            ax=ax,
            fig=fig,
            figsize=(9, 5),
        )
        self._perform_plot(ax, **kwargs)

        plt.grid()
        plt.ylabel("Dec")
        plt.xlabel("RA")
        plt.title(title)
        return fig, ax

    def _perform_plot(self, ax: WCSAxes, **kwargs):
        """Perform the plot of the search region on an initialized WCSAxes"""
        raise NotImplementedError("Plotting has not been implemented for this search")
