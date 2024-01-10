from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
from hipscat.pixel_math.polygon_filter import SphericalCoordinates


class CoordinatesValidator:
    """Performs right ascension / declination coordinate validation"""

    @staticmethod
    def validate_cone_search_params(ra: float, dec: float, radius: float) -> float:
        """Checks that the cone center has valid declination and wraps the value
        for its right ascension. It also makes sure the cone radius is positive.

        Arguments:
            ra (float): The right ascension to the center of the cone
            dec (float): The declination to the center of the cone
            radius (float): The cone radius, in degrees

        Returns:
            The wrapped value for the right ascension of the cone center, in the
            [0,360] degrees range.
        """
        if radius < 0:
            raise ValueError("Cone radius must be non negative")
        CoordinatesValidator._validate_dec_values(dec)
        wrapped_ra = CoordinatesValidator._wrap_ra_values(ra).flat[0]
        return float(wrapped_ra)

    @staticmethod
    def validate_polygon_search_params(vertices: List[SphericalCoordinates]) -> List[SphericalCoordinates]:
        """Checks that the polygon vertices have valid declination and wraps the values
         for their right ascensions.

        Arguments:
            vertices (List[Tuple[float, float]): The list of vertices of the polygon to
                filter pixels with, as a list of (ra,dec) coordinates, in degrees.

        Returns:
            The list of spherical coordinates for the polygon vertices with the wrapped
            values for the right ascension, in the [0,360] degrees range.
        """
        ra, dec = np.array(vertices).T
        CoordinatesValidator._validate_dec_values(dec)
        wrapped_ra = CoordinatesValidator._wrap_ra_values(ra)
        return list(zip(wrapped_ra, dec))

    @staticmethod
    def _validate_dec_values(dec: float | List[float]):
        """Checks that declination angles are in the [-90,90] degrees range"""
        dec_values = np.array(dec)
        if not np.all((dec_values >= -90) & (dec_values <= 90)):
            raise ValueError("dec must be between -90 and 90")

    @staticmethod
    def _wrap_ra_values(ra: float | List[float]) -> npt.NDArray[np.float64]:
        """Wraps right ascension angles to the [0,360] degrees range"""
        wrap_angle = 360.
        ra_values = np.array(ra, dtype=float)
        wrap_quotients = ra_values // wrap_angle
        if np.any(wrap_quotients != 0):
            ra_values -= wrap_quotients * wrap_angle
            ra_values[ra_values >= wrap_angle] -= wrap_angle
            ra_values[ra_values < 0] += wrap_angle
        return ra_values
