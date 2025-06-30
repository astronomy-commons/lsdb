from lsdb.core.source_association.baseline_source_associator import BaselineSourceAssociationAlgorithm


def test_source_assoc(small_sky_source_catalog):
    associator = BaselineSourceAssociationAlgorithm()
    cat = small_sky_source_catalog.associate_sources(associator, object_id_column_name="new_obj_id")
    pass
