import matplotlib.pyplot as plt
from typing import Tuple, Type, List

import astropy
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
    fov: Quantity | Tuple[Quantity, Quantity] = None,
    center: SkyCoord | None = None,
    wcs: astropy.wcs.WCS = None,
    frame_class: Type[BaseFrame] | None = None,
    ax: WCSAxes | None = None,
    fig: Figure | None = None,
    **kwargs,
):
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
        plt.colorbar(collection)

    return fig, ax
