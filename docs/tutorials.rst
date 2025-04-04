Tutorials
========================================================================================

These pages contain a set of tutorial notebooks for working through core and more advanced LSDB
functionality.

TODO - numbering is weird.

1 - LSDB objects
------------------------------------------------------

An introduction to LSDB's core features

.. toctree::
    :maxdepth: 1

    Catalog <todo>
    NestedFrame <todo>
    Margins <tutorials/margins>
    Dask Client <todo>

2 - Working with the objects
------------------------------------------------------

An introduction to the most common Catalog operations

.. toctree::
    :maxdepth: 1

    Getting data into LSDB  <tutorials/getting_data>
    Region search filtering (e.g. search_filter) <todo>
    Column filtering (e.g. columns=) <tutorials/filtering_large_catalogs>
    Complex filtering (e.g. .query) <todo>
    Crossmatching catalogs <todo>
    Applying a function (e.g. map_partitions) <tutorials/pre_executed/map_partitions>
    Working with TimeSeries <todo>
    Plotting results <tutorials/pre_executed/plotting>
    Exporting results (e.g. to_hats) <tutorials/exporting_results>

3 - Less common operations
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    Joining catalogs <tutorials/pre_executed/join_catalogs>

4 - HATS creation and reading
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    Import catalogs <tutorials/import_catalogs>
    Manual catalog verification <tutorials/pre_executed/manual_verification>
    Accessing remote data <tutorials/remote_data>
    Troubleshooting remote access <todo>

5 - Performance Tips
------------------------------------------------------

.. toctree::
    :maxdepth: 1

    Index tables <tutorials/pre_executed/index_table>
    Dask cluster configuration <tutorials/dask-cluster-tips>
    Performance testing <tutorials/performance>

6 - Science Examples
------------------------------------------------------

Notebooks going over specific, contributed scientific example use cases

.. toctree::
    :maxdepth: 1

    Cross-match ZTF BTS and NGC <tutorials/pre_executed/ztf_bts-ngc>
    Import and cross-match DES and Gaia <tutorials/pre_executed/des-gaia>
    Get a list of light-curves from ZTF and PS1 <tutorials/pre_executed/zubercal-ps1-snad>
    Search for Supernovae in ZTF alerts <tutorials/pre_executed/ztf-alerts-sne>