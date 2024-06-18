from typing import Dict, List, Tuple

from hipscat.pixel_math import HealpixPixel
from typing_extensions import TypeAlias

# Compute pixel map returns a tuple. The first element is
# the number of data points within the HEALPix pixel, the
# second element is the list of pixels it contains.
HealpixInfo: TypeAlias = Tuple[int, List[int]]

DaskDFPixelMap = Dict[HealpixPixel, int]
