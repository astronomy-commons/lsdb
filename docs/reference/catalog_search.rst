Structured Search
~~~~~~~~~~~~~~~~~~

A structured search on an LSDB catalog has two components:

- a coarse filter to limit **which** data partitions will be loaded
- a fine filter to limit the rows **inside** the data partitions

Most commonly, these are region-based filters, since the partitions
can be easily determined for specific regions on the sphere.
Culling data according to some shared region is a good way to limit
the data you're loading, while prototyping some full-sky pipeline.

There are a few ways to execute a structured search on LSDB catalogs.

The preferred approach is at the time of loading a catalog. 
You can instantiate some ``search_object`` and pass to the open call, with 
something like: ``lsdb.open_catalog(path, search_filter=search_object)``

We provide several built-in search types:

.. currentmodule:: lsdb.core.search.region_search

.. autosummary::
    :toctree: api/

    ConeSearch
    BoxSearch
    PolygonSearch
    OrderSearch
    PixelSearch
    MOCSearch

.. currentmodule:: lsdb.core.search.index_search

.. autosummary::
    :toctree: api/

    IndexSearch

Alternatively, you can call the ``search`` method after a catalog has been
opened and other operations have been performed with ``catalog.search(search_object)``.
You can also call region search methods on the catalog object.

To see region searches in action, check out :doc:`/tutorials/region_selection`.

To see an example of defining your own type of structured search, see
:doc:`/tutorials/pre_executed/custom_search`.

.. currentmodule:: lsdb.catalog

.. autosummary::
    :toctree: api/

    Catalog.cone_search
    Catalog.box_search
    Catalog.polygon_search
    Catalog.id_search
    Catalog.order_search
    Catalog.pixel_search
    Catalog.moc_search
    Catalog.search


