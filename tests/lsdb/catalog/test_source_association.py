from collections import Counter

import numpy as np

from lsdb.core.source_association.baseline_source_associator import BaselineSourceAssociationAlgorithm

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from lsdb.core.source_association.basic_object_aggregator import BasicObjectAggregator


def test_source_assoc(small_sky_source_catalog):
    # small_sky_source_catalog.plot_points()
    # plt.show()
    print(
        max(
            small_sky_source_catalog.compute()
            .groupby("exposure")
            .apply(lambda x: len(x) - len(np.unique(x["object_id"])))
        )
    )
    associator = BaselineSourceAssociationAlgorithm(exposure_id_col="exposure", max_distance=3100)
    cat = small_sky_source_catalog.associate_sources(
        associator, "source_id", object_id_column_name="new_obj_id"
    )
    aggregator = BasicObjectAggregator(small_sky_source_catalog.hc_structure.catalog_info)
    obj = small_sky_source_catalog.associate_sources(
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
    # cat.plot_pixels(fc="#00000000", ec="black", color_by_order=False)
    # cat.plot_points(color_col="new_obj_id", marker="+")
    # # cat.plot_points(color_col="object_id", marker="x")
    # plt.show()
    # a = cat.compute()
    # # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true=a["object_id"], y_pred=a["new_obj_id"]))
    # # # Then just plot it:
    # # disp.plot()
    # # # And show it:
    # # plt.show()
    # b = a.groupby("object_id").apply(lambda x: Counter(x["new_obj_id"]).most_common(1)[0][0])
    # value_to_index = {v: k for k, v in b.items()}
    # result_obj_id = np.array([value_to_index.get(x, x) for x in a["new_obj_id"]])
    # print(f1_score(y_true=a["object_id"], y_pred=result_obj_id, average="micro"))
    # # disp = ConfusionMatrixDisplay(
    # #     confusion_matrix=confusion_matrix(y_true=a["object_id"], y_pred=result_obj_id)
    # # )
    # # # Then just plot it:
    # # disp.plot()
    # # # And show it:
    # # plt.show()
    # pass
