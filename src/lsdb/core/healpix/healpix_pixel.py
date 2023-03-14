from __future__ import annotations

MAXIMUM_ORDER = 29


class HealpixPixel:
    """A HEALPix pixel, represented by an order and pixel number in NESTED ordering scheme"""

    def __init__(self, order: int, pixel: int) -> None:
        """Initialize a HEALPix pixel
        Args:
            order: HEALPix order
            pixel: HEALPix pixel number in NESTED ordering scheme
        """
        if order > MAXIMUM_ORDER:
            raise ValueError(f"HEALPix order cannot be greater than {MAXIMUM_ORDER}")
        self.order = order
        self.pixel = pixel

    def _key(self) -> tuple[int, int]:
        """Returns tuple of order and pixel, for use in hashing and equality"""
        return self.order, self.pixel

    def __eq__(self, other: HealpixPixel) -> bool:
        """Defines 2 pixels as equal if they have the same order and pixel"""
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hashes pixels by order and pixel, so equal pixel objects are looked up the same in
        hashable data structures"""
        return hash(self._key())
