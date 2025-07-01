from idlelib.rpc import objecttable

import nested_pandas as npd
import numpy as np
from dill import objects
from hats import HealpixPixel
from hats.catalog import TableProperties
from numpy.f2py.crackfortran import sourcecodeform

from lsdb.core.crossmatch.kdtree_utils import _lon_lat_to_xyz, _find_crossmatch_indices, _get_chord_distance
from lsdb.core.source_association.abstract_source_association_algorithm import (
    AbstractSourceAssociationAlgorithm,
)


class BaselineSourceAssociationAlgorithm(AbstractSourceAssociationAlgorithm):

    def __init__(self, exposure_id_col: str, max_distance: float):
        self.exposure_id_col = exposure_id_col
        self.max_distance = max_distance

    def associate_sources(
        self,
        partition: npd.NestedFrame,
        pixel: HealpixPixel,
        properties: TableProperties,
        margin_properties: TableProperties,
        source_id_col: str,
    ) -> np.ndarray:
        df = partition.sort_values(by=[self.exposure_id_col])
        exposures = df[self.exposure_id_col].to_numpy()
        unique_exposures = np.unique(exposures)
        xyz = _lon_lat_to_xyz(
            lon=df[properties.ra_column].to_numpy(),
            lat=df[properties.dec_column].to_numpy(),
        )
        s_ids = df[source_id_col].to_numpy()
        max_dist_chord = _get_chord_distance(self.max_distance)
        object_id_assignment = np.full(s_ids.shape, fill_value=-1)
        for e_id in unique_exposures:
            mask = exposures == e_id
            if np.all(object_id_assignment == -1):
                object_id_assignment[mask] = s_ids[mask]
                continue
            object_ids, object_inds = np.unique(object_id_assignment, return_index=True)
            unique_inds = object_inds[np.where(object_ids != -1)[0]]
            object_xyz = xyz[unique_inds]
            source_xyz = xyz[mask]
            _, source_idx, object_idx = _find_crossmatch_indices(
                source_xyz, object_xyz, n_neighbors=1, max_distance=max_dist_chord
            )
            if len(object_idx) > 0:
                matched_source_inds = np.where(mask)[0][source_idx]
                object_inds_for_matches = object_id_assignment[unique_inds[object_idx]]
                object_id_assignment[matched_source_inds] = object_inds_for_matches
            remaining_source_idx = np.delete(np.arange(len(source_xyz)), source_idx)
            remaining_source_inds = np.where(mask)[0][remaining_source_idx]
            object_id_assignment[remaining_source_inds] = s_ids[remaining_source_inds]
        return object_id_assignment
