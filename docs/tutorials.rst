Tutorials
========================================================================================

These pages contain a set of tutorial notebooks for working through core and more advanced LSDB
functionality.

1 - Catalogs
------------------------------------------------------

An introduction to LSDB's core feature: ``Catalog``

.. toctree::
    :maxdepth: 1

    The Catalog Object <tutorials/catalog_object>
    Getting data into LSDB  <tutorials/getting_data>
    Column filtering (e.g. columns=) <tutorials/column_filtering>
    Region selection filtering (e.g. search_filter) <tutorials/region_selection>
    Row filtering (e.g. .query) <tutorials/row_filtering>
    Single row selection <tutorials/pre_executed/index_table>

2 - Analyzing Catalogs
------------------------------------------------------

An introduction to the most common Catalog operations

.. toctree::
    :maxdepth: 1

    Setting up a Dask Client <tutorials/dask_client>
    Margins <tutorials/margins>
    Crossmatching catalogs <tutorials/pre_executed/crossmatching>
    Applying a function (e.g. map_partitions) <tutorials/pre_executed/map_partitions>
    Working with TimeSeries <tutorials/pre_executed/timeseries>
    Plotting results <tutorials/pre_executed/plotting>
    Exporting results (e.g. to_hats) <tutorials/exporting_results>

3 - Nested data manipulation
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    NestedFrame <tutorials/nested_frame>

4 - HATS creation and reading
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    Import catalogs <tutorials/import_catalogs>
    Manual catalog verification <tutorials/pre_executed/manual_verification>
    Accessing remote data <tutorials/remote_data>

5 - Performance Tips
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    Index tables <tutorials/pre_executed/index_table>
    Dask cluster configuration <tutorials/dask-cluster-tips>
    Performance testing <tutorials/performance>
    Joining catalogs <tutorials/pre_executed/join_catalogs>

6 - Science Examples
------------------------------------------------------

Notebooks going over specific, contributed scientific example use cases

.. toctree::
    :maxdepth: 1

    Cross-match ZTF BTS and NGC <tutorials/pre_executed/ztf_bts-ngc>
    Import and cross-match DES and Gaia <tutorials/pre_executed/des-gaia>
    Get a list of light-curves from ZTF and PS1 <tutorials/pre_executed/zubercal-ps1-snad>
    Search for Supernovae in ZTF alerts <tutorials/pre_executed/ztf-alerts-sne>
    Working with rubin data preview 1 <tutorials/rubin_dp1>
