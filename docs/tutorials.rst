Tutorials
========================================================================================

These tutorials cover core and advanced LSDB functionality organized by task.
They can be read in any order — if you're just starting out, see :doc:`getting-started` first.

Filtering and Selecting Data
----------------------------

The catalog object, lazy evaluation, and all the ways to narrow down what you're working with.

.. toctree::
   :maxdepth: 1

   The Catalog Object </tutorials/catalog_object>
   Lazy Operations in LSDB </tutorials/lazy_operations>
   Column filtering (e.g. columns=) </tutorials/column_filtering>
   Row filtering (e.g. .query) </tutorials/row_filtering>
   Region selection (e.g. search_filter) </tutorials/region_selection>
   Margin Catalogs </tutorials/margins>
   Custom Structured Search </tutorials/pre_executed/custom_search>

.. note::

   Coming soon: *Developing at Small Scale* — a guide to working with data subsets
   during development before scaling to the full catalog.

Combining Catalogs
------------------

Spatial crossmatching, identifier-based joins, and index-accelerated lookups.

.. toctree::
   :maxdepth: 1

   Crossmatching Catalogs </tutorials/pre_executed/crossmatching>
   Joining Catalogs </tutorials/pre_executed/join_catalogs>
   Using Index Tables </tutorials/pre_executed/index_table>

Nested and Time-Series Data
---------------------------

Working with nested-pandas ``NestedFrame`` objects and light-curve data.

.. toctree::
   :maxdepth: 1

   Understanding the NestedFrame </tutorials/pre_executed/nestedframe>
   Working with Time Series Data </tutorials/pre_executed/timeseries>
   Exploding Lightcurves to Flat Source Tables </tutorials/pre_executed/explode_lightcurves>

Importing and Exporting
-----------------------

Converting data to HATS format, validating catalogs, and saving results.

.. toctree::
   :maxdepth: 1

   Importing Catalogs to HATS Format </tutorials/import_catalogs>
   Manual Catalog Verification </tutorials/pre_executed/manual_verification>
   Exporting Results </tutorials/exporting_results>

Infrastructure and Performance
------------------------------

Dask setup, distributed computing, profiling, and visualization.

.. toctree::
   :maxdepth: 1

   Setting up a Dask Client </tutorials/dask_client>
   Applying a Function with map_partitions </tutorials/pre_executed/map_partitions>
   Scaling Workflows </tutorials/pre_executed/scaling_workflows>
   Dask Cluster Configuration Tips </tutorials/dask-cluster-tips>
   Kubernetes Deployment </tutorials/kubernetes-deployment>
   Troubleshooting Dask Messages </tutorials/dask-messages-guide>
   Performance Benchmarks </tutorials/performance>
   Plotting Results </tutorials/pre_executed/plotting>
