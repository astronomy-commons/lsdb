from __future__ import annotations

from typing import Tuple, Type

import astropy
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import BaseFrame
from hats.inspection.visualize_catalog import initialize_wcs_axes
from matplotlib.figure import Figure
from mocpy.moc.plot.utils import _set_wcs


def plot_points(
    df: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    *,
    color_col: str | None = None,
    projection: str = "MOL",
    title: str = "",
    fov: Quantity | Tuple[Quantity, Quantity] | None = None,
    center: SkyCoord | None = None,
    wcs: astropy.wcs.WCS | None = None,
    frame_class: Type[BaseFrame] | None = None,
    ax: WCSAxes | None = None,
    fig: Figure | None = None,
    **kwargs,
):
    """Plots the points in a given dataframe as a scatter plot

    Performs a scatter plot on a WCSAxes with the points in a dataframe.
    The scatter points can be colored by a column of the catalog by using the `color_col` kwarg

    Args:
        ra_column (str | None): The column to use as the RA of the points to plot. Defaults to the
            catalog's default RA column. Useful for plotting joined or cross-matched points
        dec_column (str | None): The column to use as the Declination of the points to plot. Defaults to
            the catalog's default Declination column. Useful for plotting joined or cross-matched points
        color_col (str | None): The column to use as the color array for the scatter plot. Allows coloring
            of the points by the values of a given column.
        projection (str): The projection to use in the WCS. Available projections listed at
            https://docs.astropy.org/en/stable/wcs/supported_projections.html
        title (str): The title of the plot
        fov (Quantity or Sequence[Quantity, Quantity] | None): The Field of View of the WCS. Must be an
            astropy Quantity with an angular unit, or a tuple of quantities for different longitude and \
            latitude FOVs (Default covers the full sky)
        center (SkyCoord | None): The center of the projection in the WCS (Default: SkyCoord(0, 0))
        wcs (WCS | None): The WCS to specify the projection of the plot. If used, all other WCS parameters
            are ignored and the parameters from the WCS object is used.
        frame_class (Type[BaseFrame] | None): The class of the frame for the WCSAxes to be initialized with.
            if the `ax` kwarg is used, this value is ignored (By Default uses EllipticalFrame for full
            sky projection. If FOV is set, RectangularFrame is used)
        ax (WCSAxes | None): The matplotlib axes to plot onto. If None, an axes will be created to be used. If
            specified, the axes must be an astropy WCSAxes, and the `wcs` parameter must be set with the WCS
            object used in the axes. (Default: None)
        fig (Figure | None): The matplotlib figure to add the axes to. If None, one will be created, unless
            ax is specified (Default: None)
        **kwargs: Additional kwargs to pass to creating the matplotlib `scatter` function. These include
            `c` for color, `s` for the size of hte points, `marker` for the maker type, `cmap` and `norm`
            if `color_col` is used

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

    ra = df[ra_column].to_numpy()
    dec = df[dec_column].to_numpy()
    if color_col is not None:
        kwargs["c"] = df[color_col].to_numpy()
    collection = None
    if len(ra) > 0:
        collection = ax.scatter(ra, dec, transform=ax.get_transform("icrs"), **kwargs)

    # Set projection
    _set_wcs(ax, wcs)

    ax.coords[0].set_format_unit("deg")

    plt.grid()
    plt.ylabel("Dec")
    plt.xlabel("RA")
    plt.title(title)
    if color_col is not None and collection is not None:
        plt.colorbar(collection, label=color_col)
    return fig, ax
