import nested_pandas as npd
import numpy as np
from hats import HealpixPixel
from hats.catalog import TableProperties

from lsdb.core.crossmatch.kdtree_utils import _find_crossmatch_indices, _get_chord_distance, _lon_lat_to_xyz
from lsdb.core.source_association.abstract_source_association_algorithm import (
    AbstractSourceAssociationAlgorithm,
)


class BaselineSourceAssociationAlgorithm(AbstractSourceAssociationAlgorithm):
    def __init__(self, exposure_id_col: str, max_distance_arcsec: float):
        self.exposure_id_col = exposure_id_col
        self.max_distance_arcsec = max_distance_arcsec

    def associate_sources(
        self,
        partition: npd.NestedFrame,
        pixel: HealpixPixel,
        properties: TableProperties,
        margin_properties: TableProperties,
        source_id_col: str,
    ) -> np.ndarray:
        # Sort by exposures
        df = partition.sort_values(by=[self.exposure_id_col, source_id_col])
        # Get arrays of exposures, cartesian coords, source ids
        exposures = df[self.exposure_id_col].to_numpy()
        unique_exposures = np.unique(exposures)
        xyz = _lon_lat_to_xyz(
            lon=df[properties.ra_column].to_numpy(),
            lat=df[properties.dec_column].to_numpy(),
        )
        s_ids = df[source_id_col].to_numpy()
        # Get max xmatch distance as a chord
        max_dist_chord = _get_chord_distance(self.max_distance_arcsec)
        # Make array to assign object ids to sources
        object_id_assignment = np.full(s_ids.shape, fill_value=-1)
        # Make array of the objects we've found so far as the indices of the source df they correspond to
        first_object_inds = np.array([])

        for e_id in unique_exposures:
            # Find rows in this exposure
            e_mask = exposures == e_id
            if np.all(object_id_assignment == -1):
                # If first exposure, assign all as new objects (obj id is source id)
                object_id_assignment[e_mask] = s_ids[e_mask]
                object_inds = np.where(e_mask)[0]
                first_object_inds = object_inds
                continue
            # Get the coordinates of the objects we've found so far
            objects_xyz = xyz[first_object_inds]
            # Get the coordinates of the current exposure
            sources_xyz = xyz[e_mask]
            # Crossmatch the current exposure to the current objects: get indices into _xyz arrays
            xmatch_distances, xmatch_sources_idx, xmatch_objects_idx = _find_crossmatch_indices(
                sources_xyz, objects_xyz, n_neighbors=1, max_distance=max_dist_chord
            )
            # Ensure each object matches to at most one source in this exposure
            unique_object_idx, object_idx_counts = np.unique(xmatch_objects_idx, return_counts=True)
            for object_id in unique_object_idx[object_idx_counts > 1]:
                # Index into crossmatch indices
                xmatch_idx_idx = np.where(xmatch_objects_idx == object_id)[0]
                xmatch_idx_idx_dups_idx = np.argsort(xmatch_distances[xmatch_idx_idx])[1:]
                xmatch_dups_idx_idx = xmatch_idx_idx[xmatch_idx_idx_dups_idx]
                # Remove (by index) any matches that aren't the closest match
                xmatch_distances = np.delete(xmatch_distances, xmatch_dups_idx_idx)
                xmatch_sources_idx = np.delete(xmatch_sources_idx, xmatch_dups_idx_idx)
                xmatch_objects_idx = np.delete(xmatch_objects_idx, xmatch_dups_idx_idx)
            if len(xmatch_objects_idx) > 0:
                # Assign the Object ID of the objects to their matched sources
                matched_source_inds = np.where(e_mask)[0][xmatch_sources_idx]
                object_inds_for_matches = object_id_assignment[first_object_inds[xmatch_objects_idx]]
                object_id_assignment[matched_source_inds] = object_inds_for_matches
            # Assign new IDs to the unmatched sources, and add them as new objects
            remaining_source_idx = np.delete(np.arange(len(sources_xyz)), xmatch_sources_idx)
            remaining_source_inds = np.where(e_mask)[0][remaining_source_idx]
            object_id_assignment[remaining_source_inds] = s_ids[remaining_source_inds]
            first_object_inds = np.concat([first_object_inds, remaining_source_inds])
        # Assign the object id assigment and return the Pandas Series to get the right index
        df["new_obj_id"] = object_id_assignment
        return df.sort_index()["new_obj_id"]
