=========
Catalog
=========
.. currentmodule:: lsdb.catalog

Catalog Types
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog
    MarginCatalog
    MapCatalog

Properties
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog.hc_structure
    Catalog.dtypes
    Catalog.columns
    Catalog.nested_columns
    Catalog.all_columns
    Catalog.original_schema
    Catalog.margin

Inspection Methods
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog.get_healpix_pixels
    Catalog.get_ordered_healpix_pixels
    Catalog.aggregate_column_statistics
    Catalog.per_pixel_statistics
    Catalog.partitions
    Catalog.npartitions
    Catalog.plot_pixels
    Catalog.plot_coverage
    Catalog.plot_points

Search Methods
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog.iloc
    Catalog.loc
    Catalog.query
    Catalog.cone_search
    Catalog.box_search
    Catalog.polygon_search
    Catalog.id_search
    Catalog.order_search
    Catalog.pixel_search
    Catalog.moc_search
    Catalog.search
    Catalog.__getitem__
    Catalog.__len__

Dataframe Methods
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog.assign
    Catalog.dropna
    Catalog.reduce
    Catalog.sort_nested_values
    Catalog.map_partitions
    Catalog.to_hats
    Catalog.compute
    Catalog.get_partition
    Catalog.get_partition_index
    Catalog.head
    Catalog.tail
    Catalog.sample
    Catalog.random_sample
    Catalog.prune_empty_partitions
    Catalog.skymap_data
    Catalog.skymap_histogram

Inter-catalog Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Catalog.crossmatch
    Catalog.crossmatch_nested
    Catalog.merge_map
    Catalog.merge
    Catalog.merge_asof
    Catalog.join
    Catalog.join_nested
    Catalog.nest_lists
