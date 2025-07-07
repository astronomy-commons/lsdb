from collections import Counter

import numpy as np

from lsdb.core.source_association.baseline_source_associator import BaselineSourceAssociationAlgorithm

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from lsdb.core.source_association.basic_object_aggregator import BasicObjectAggregator


def test_source_assoc(small_sky_source_catalog_with_margin):
    associator = BaselineSourceAssociationAlgorithm(exposure_id_col="exposure", max_distance_arcsec=3100)
    cat = small_sky_source_catalog_with_margin.associate_sources(
        associator, "source_id", object_id_column_name="new_obj_id"
    )
    aggregator = BasicObjectAggregator(small_sky_source_catalog_with_margin.hc_structure.catalog_info)
    obj = small_sky_source_catalog_with_margin.associate_sources(
        associator, "source_id", object_aggregator=aggregator, object_id_column_name="new_obj_id"
    )
    r = cat.compute()
    unique_labels = np.unique(r["new_obj_id"])

    base_cmap = plt.cm.tab20
    base_colors = base_cmap.colors  # 10 colors
    n_base_colors = len(base_colors)

    # Create a ListedColormap with one color per unique label
    repeated_colors = [base_colors[i % n_base_colors] for i in range(len(unique_labels))]
    cmap = mcolors.ListedColormap(repeated_colors)

    # Create boundaries so that each integer maps to a color bin
    # Add ±0.5 so that integers fall in the center of their bin
    boundaries = np.array(unique_labels) - 0.5
    boundaries = np.append(boundaries, unique_labels[-1] + 0.5)

    # BoundaryNorm maps your specific integer values to 0–N indices
    norm = mcolors.BoundaryNorm(boundaries, ncolors=len(unique_labels))

    obj.plot_pixels(fc="#00000000", ec="black", color_by_order=False)
    cat.plot_points(marker="+", color_col="new_obj_id", cmap=cmap, norm=norm)
    obj.plot_points()
    plt.show()
    pass
