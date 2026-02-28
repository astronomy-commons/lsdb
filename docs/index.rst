.. lsdb documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


LSDB
========================================================================================

LSDB is a python tool for scalable analysis of large catalogs (e.g. analyzing, querying 
and/or crossmatching ~10⁹ sources). It aims to address large-scale data processing 
challenges expected from upcoming surveys, such as `LSST <https://www.lsst.org/about>`_,
`Euclid <https://www.euclid-ec.org>`_, `Roman <https://roman.gsfc.nasa.gov>`_, and the
`Schmidt Observatory System <https://www.schmidtsciences.org/schmidt-observatory-system/>`_.


Built on top of `Dask <https://docs.dask.org/>`_ to efficiently scale and parallelize operations across multiple distributed workers, it
uses the `HATS <https://hats.readthedocs.io/en/stable/>`_ data format to efficiently perform spatial
operations.

.. raw:: html

    <div class="api-surface-wrapper">
       <img src="_static/API_Surface_Feb_12.png"
               alt="LSDB API surface map with clickable function and class labels"
               usemap="#lsdb-api-surface-map"
         data-map-width="3719"
         data-map-height="2164"
         data-original-map-width="3719"
         data-original-map-height="2164"
               data-hitbox-scale-x="1.0"
               data-hitbox-scale-y="1.0"
               class="api-surface-image" />
       <map name="lsdb-api-surface-map">
          <area shape="rect" coords="126,178,549,246" href="reference/api/lsdb.open_catalog.html" alt="open_catalog" />
          <area shape="rect" coords="126,238,549,306" href="reference/api/lsdb.from_dataframe.html" alt="from_dataframe" />
          <area shape="rect" coords="126,298,549,366" href="reference/api/lsdb.from_astropy.html" alt="from_astropy" />
          <area shape="rect" coords="126,358,549,426" href="reference/api/lsdb.nested.datasets.generation.generate_catalog.html" alt="generate_catalog" />
          <area shape="rect" coords="417,632,721,720" href="reference/api/lsdb.show_versions.html" alt="show_versions" />
          <area shape="rect" coords="487,802,829,870" href="reference/api/lsdb.catalog.Catalog.plot_pixels.html" alt="plot_pixels" />
          <area shape="rect" coords="487,862,829,930" href="reference/api/lsdb.catalog.Catalog.plot_coverage.html" alt="plot_coverage" />
          <area shape="rect" coords="498,967,782,1036" href="reference/api/lsdb.catalog.Catalog.plot_points.html" alt="plot_points" />

          <area shape="rect" coords="993,228,1743,296" href="reference/api/lsdb.catalog.Catalog.html" alt="Catalog" />
          <area shape="rect" coords="1010,306,1576,374" href="reference/catalog_properties.html" alt="name" />
          <area shape="rect" coords="1010,366,1218,434" href="reference/api/lsdb.catalog.Catalog.columns.html" alt="columns" />
          <area shape="rect" coords="1257,366,1576,434" href="reference/api/lsdb.catalog.Catalog.all_columns.html" alt="all_columns" />
          <area shape="rect" coords="1010,426,1539,494" href="reference/api/lsdb.catalog.Catalog.nested_columns.html" alt="nested_columns" />
          <area shape="rect" coords="1010,486,1384,555" href="reference/api/lsdb.catalog.Catalog.original_schema.html" alt="original_schema" />
          <area shape="rect" coords="1416,486,1576,555" href="reference/api/lsdb.catalog.Catalog.dtypes.html" alt="dtypes" />
          <area shape="rect" coords="1010,590,1223,654" href="reference/api/lsdb.catalog.Catalog.head.html" alt="head" />
          <area shape="rect" coords="1306,590,1520,654" href="reference/api/lsdb.catalog.Catalog.tail.html" alt="tail" />
          <area shape="rect" coords="1010,646,1181,709" href="reference/api/lsdb.catalog.Catalog.sample.html" alt="sample" />
          <area shape="rect" coords="1217,646,1569,709" href="reference/api/lsdb.catalog.Catalog.random_sample.html" alt="random_sample" />
          <area shape="rect" coords="1016,740,1188,809" href="reference/api/lsdb.catalog.Catalog.rename.html" alt="rename" />
          <area shape="rect" coords="1826,269,2147,338" href="reference/api/lsdb.catalog.Catalog.hc_structure.html" alt="hc_structure" />
          <area shape="rect" coords="1817,399,2135,467" href="reference/catalog_properties.html" alt="hc_collection" />
          <area shape="rect" coords="1770,425,2201,708" href="reference/api/lsdb.catalog.MarginCatalog.html" alt="MarginCatalog" />
          <area shape="rect" coords="1805,649,2169,748" href="reference/api/lsdb.catalog.MapCatalog.html" alt="MapCatalog" />
         <area shape="rect" coords="1763,793,2212,859" href="reference/api/lsdb.io.to_association.html" alt="AssociationCatalog" />

          <area shape="rect" coords="1233,948,1308,1018" href="reference/api/lsdb.catalog.Catalog.__len__.html" alt="len" />
          <area shape="rect" coords="974,1044,1664,1121" href="reference/api/lsdb.catalog.Catalog.aggregate_column_statistics.html" alt="aggregate_column_statistics" />
          <area shape="rect" coords="1697,1044,2212,1121" href="reference/api/lsdb.catalog.Catalog.per_pixel_statistics.html" alt="per_pixel_statistics" />
          <area shape="rect" coords="974,1148,1451,1218" href="reference/api/lsdb.catalog.Catalog.get_healpix_pixels.html" alt="get_healpix_pixels" />
          <area shape="rect" coords="1486,1148,2169,1218" href="reference/api/lsdb.catalog.Catalog.get_ordered_healpix_pixels.html" alt="get_ordered_healpix_pixels" />
          <area shape="rect" coords="974,1210,1258,1279" href="reference/api/lsdb.catalog.Catalog.partitions.html" alt="partitions" />
          <area shape="rect" coords="1295,1210,1606,1279" href="reference/api/lsdb.catalog.Catalog.npartitions.html" alt="npartitions" />
          <area shape="rect" coords="1644,1210,2169,1279" href="reference/api/lsdb.catalog.Catalog.get_partition_index.html" alt="get_partition_index" />
          <area shape="rect" coords="981,1319,1333,1388" href="reference/api/lsdb.catalog.Catalog.estimate_size.html" alt="estimate_size" />
          <area shape="rect" coords="1397,1324,1730,1392" href="reference/api/lsdb.catalog.Catalog.get_partition.html" alt="get_partition" />

          <area shape="rect" coords="1022,1593,1593,1661" href="reference/api/lsdb.catalog.Catalog.prune_empty_partitions.html" alt="prune_empty_partitions" />
          <area shape="rect" coords="1651,1486,2108,1580" href="reference/api/lsdb.catalog.Catalog.map_partitions.html" alt="map_partitions" />
          <area shape="rect" coords="1651,1572,2108,1666" href="reference/api/lsdb.catalog.Catalog.map_rows.html" alt="map_rows" />
          <area shape="rect" coords="1046,1697,1556,1774" href="reference/api/lsdb.catalog.Catalog.to_dask_dataframe.html" alt="to_dask_dataframe" />
          <area shape="rect" coords="1046,1766,1556,1843" href="reference/api/lsdb.catalog.Catalog.to_delayed.html" alt="to_delayed" />
          <area shape="rect" coords="1397,1324,1730,1392" href="reference/api/lsdb.catalog.Catalog.get_partition.html" alt="get_partition execution" />

          <area shape="rect" coords="2510,342,2650,412" href="reference/api/lsdb.catalog.Catalog.query.html" alt="query" />
          <area shape="rect" coords="2497,454,2834,517" href="reference/api/lsdb.catalog.Catalog.box_search.html" alt="box_search" />
          <area shape="rect" coords="2497,509,2834,571" href="reference/api/lsdb.catalog.Catalog.cone_search.html" alt="cone_search" />
          <area shape="rect" coords="2497,563,2834,625" href="reference/api/lsdb.catalog.Catalog.polygon_search.html" alt="polygon_search" />
          <area shape="rect" coords="2492,665,2815,733" href="reference/api/lsdb.catalog.Catalog.order_search.html" alt="order_search" />
          <area shape="rect" coords="2492,725,2815,793" href="reference/api/lsdb.catalog.Catalog.pixel_search.html" alt="pixel_search" />
          <area shape="rect" coords="2508,823,2748,891" href="reference/api/lsdb.catalog.Catalog.id_search.html" alt="id_search" />
          <area shape="rect" coords="2507,917,2676,987" href="reference/api/lsdb.catalog.Catalog.search.html" alt="search" />
          <area shape="rect" coords="2504,1019,2788,1087" href="reference/api/lsdb.catalog.Catalog.moc_search.html" alt="moc_search" />

          <area shape="rect" coords="3097,230,3448,384" href="reference/api/lsdb.catalog.Catalog.crossmatch.html" alt="crossmatch" />
          <area shape="rect" coords="3106,432,3576,500" href="reference/api/lsdb.catalog.Catalog.crossmatch_nested.html" alt="crossmatch_nested" />
          <area shape="rect" coords="3120,693,3331,761" href="reference/api/lsdb.catalog.Catalog.merge.html" alt="merge" />
          <area shape="rect" coords="3120,753,3409,821" href="reference/api/lsdb.catalog.Catalog.concat.html" alt="concat" />
          <area shape="rect" coords="3080,606,3225,690" href="reference/api/lsdb.catalog.Catalog.join.html" alt="join" />
          <area shape="rect" coords="3120,933,3409,1001" href="reference/api/lsdb.catalog.Catalog.join_nested.html" alt="join_nested" />
          <area shape="rect" coords="3120,1054,3409,1122" href="reference/api/lsdb.catalog.Catalog.nest_lists.html" alt="nest_lists" />
          <area shape="rect" coords="3266,1247,3549,1315" href="reference/api/lsdb.catalog.Catalog.merge_asof.html" alt="merge_asof" />
          <area shape="rect" coords="3266,1307,3549,1375" href="reference/api/lsdb.catalog.Catalog.merge_map.html" alt="merge_map" />

          <area shape="rect" coords="2513,1346,2832,1407" href="reference/api/lsdb.streams.CatalogStream.html" alt="CatalogStream" />
          <area shape="rect" coords="2513,1399,2832,1461" href="reference/api/lsdb.streams.InfiniteStream.html" alt="InfiniteStream" />

          <area shape="rect" coords="1761,1906,2109,2019" href="reference/api/lsdb.catalog.Catalog.compute.html" alt="compute" />
          <area shape="rect" coords="2244,1791,2808,2141" href="reference/api/lsdb.catalog.Catalog.write_catalog.html" alt="write_catalog" />
          <area shape="rect" coords="2902,1822,3113,1890" href="reference/api/lsdb.io.to_hats.html" alt="to_hats" />
          <area shape="rect" coords="3177,1824,3546,1892" href="reference/api/lsdb.io.to_association.html" alt="to_association" />
       </map>
    </div>

.. rubric:: API surface quick links (accessible fallback)

- Data Ingress: `open_catalog <reference/api/lsdb.open_catalog.html>`_, `from_dataframe <reference/api/lsdb.from_dataframe.html>`_, `from_astropy <reference/api/lsdb.from_astropy.html>`_, `generate_catalog <reference/api/lsdb.nested.datasets.generation.generate_catalog.html>`_
- Utilities / Visualization: `show_versions <reference/api/lsdb.show_versions.html>`_, `plot_pixels <reference/api/lsdb.catalog.Catalog.plot_pixels.html>`_, `plot_coverage <reference/api/lsdb.catalog.Catalog.plot_coverage.html>`_, `plot_points <reference/api/lsdb.catalog.Catalog.plot_points.html>`_
- Catalog Types: `Catalog <reference/api/lsdb.catalog.Catalog.html>`_, `MarginCatalog <reference/api/lsdb.catalog.MarginCatalog.html>`_, `MapCatalog <reference/api/lsdb.catalog.MapCatalog.html>`_, `AssociationCatalog <reference/api/lsdb.io.to_association.html>`_
- Catalog Properties: `hc_structure <reference/api/lsdb.catalog.Catalog.hc_structure.html>`_, `hc_collection <reference/catalog_properties.html>`_, `columns <reference/api/lsdb.catalog.Catalog.columns.html>`_, `all_columns <reference/api/lsdb.catalog.Catalog.all_columns.html>`_, `nested_columns <reference/api/lsdb.catalog.Catalog.nested_columns.html>`_, `original_schema <reference/api/lsdb.catalog.Catalog.original_schema.html>`_, `dtypes <reference/api/lsdb.catalog.Catalog.dtypes.html>`_
- Filtering / Selection: `query <reference/api/lsdb.catalog.Catalog.query.html>`_, `head <reference/api/lsdb.catalog.Catalog.head.html>`_, `tail <reference/api/lsdb.catalog.Catalog.tail.html>`_, `sample <reference/api/lsdb.catalog.Catalog.sample.html>`_, `random_sample <reference/api/lsdb.catalog.Catalog.random_sample.html>`_, `rename <reference/api/lsdb.catalog.Catalog.rename.html>`_
- Search: `box_search <reference/api/lsdb.catalog.Catalog.box_search.html>`_, `cone_search <reference/api/lsdb.catalog.Catalog.cone_search.html>`_, `polygon_search <reference/api/lsdb.catalog.Catalog.polygon_search.html>`_, `order_search <reference/api/lsdb.catalog.Catalog.order_search.html>`_, `pixel_search <reference/api/lsdb.catalog.Catalog.pixel_search.html>`_, `id_search <reference/api/lsdb.catalog.Catalog.id_search.html>`_, `search <reference/api/lsdb.catalog.Catalog.search.html>`_, `moc_search <reference/api/lsdb.catalog.Catalog.moc_search.html>`_
- Inspection: `len <reference/api/lsdb.catalog.Catalog.__len__.html>`_, `aggregate_column_statistics <reference/api/lsdb.catalog.Catalog.aggregate_column_statistics.html>`_, `per_pixel_statistics <reference/api/lsdb.catalog.Catalog.per_pixel_statistics.html>`_, `get_healpix_pixels <reference/api/lsdb.catalog.Catalog.get_healpix_pixels.html>`_, `get_ordered_healpix_pixels <reference/api/lsdb.catalog.Catalog.get_ordered_healpix_pixels.html>`_, `partitions <reference/api/lsdb.catalog.Catalog.partitions.html>`_, `npartitions <reference/api/lsdb.catalog.Catalog.npartitions.html>`_, `get_partition_index <reference/api/lsdb.catalog.Catalog.get_partition_index.html>`_, `estimate_size <reference/api/lsdb.catalog.Catalog.estimate_size.html>`_, `get_partition <reference/api/lsdb.catalog.Catalog.get_partition.html>`_
- Execution: `prune_empty_partitions <reference/api/lsdb.catalog.Catalog.prune_empty_partitions.html>`_, `map_partitions <reference/api/lsdb.catalog.Catalog.map_partitions.html>`_, `map_rows <reference/api/lsdb.catalog.Catalog.map_rows.html>`_, `to_dask_dataframe <reference/api/lsdb.catalog.Catalog.to_dask_dataframe.html>`_, `to_delayed <reference/api/lsdb.catalog.Catalog.to_delayed.html>`_, `compute <reference/api/lsdb.catalog.Catalog.compute.html>`_, `write_catalog <reference/api/lsdb.catalog.Catalog.write_catalog.html>`_
- Inter-catalog / Streaming / I/O: `crossmatch <reference/api/lsdb.catalog.Catalog.crossmatch.html>`_, `crossmatch_nested <reference/api/lsdb.catalog.Catalog.crossmatch_nested.html>`_, `merge <reference/api/lsdb.catalog.Catalog.merge.html>`_, `concat <reference/api/lsdb.catalog.Catalog.concat.html>`_, `join <reference/api/lsdb.catalog.Catalog.join.html>`_, `join_nested <reference/api/lsdb.catalog.Catalog.join_nested.html>`_, `nest_lists <reference/api/lsdb.catalog.Catalog.nest_lists.html>`_, `merge_asof <reference/api/lsdb.catalog.Catalog.merge_asof.html>`_, `merge_map <reference/api/lsdb.catalog.Catalog.merge_map.html>`_, `CatalogStream <reference/api/lsdb.streams.CatalogStream.html>`_, `InfiniteStream <reference/api/lsdb.streams.InfiniteStream.html>`_, `to_hats <reference/api/lsdb.io.to_hats.html>`_, `to_association <reference/api/lsdb.io.to_association.html>`_


Using this Guide
-------------------------------------------------------------------------------
.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
       :link: getting-started
       :link-type: doc

       Installation and QuickStart Guide

   .. grid-item-card:: Tutorials
       :link: tutorials
       :link-type: doc

       Learn the LSDB features by working through our guides

.. grid:: 1 1 1 1

   .. grid-item-card:: Data Access
       :link: data-access
       :link-type: doc

       How LSDB catalogs are served and accessed across providers

.. grid:: 1 1 2 2

   .. grid-item-card:: API Reference
       :link: reference
       :link-type: doc

       The detailed API documentation

   .. grid-item-card:: Contact Us / Getting Help
      :link: contact
      :link-type: doc

      Reach out for more help

.. toctree::
   :hidden:
   :caption: Using LSDB

   Home page <self>
   Getting Started <getting-started>
   Tutorials <tutorials>
   Data Access <data-access>
   API Reference <reference>

.. toctree::
   :hidden:
   :caption: Project

   About & Citation <citation>
   Contact Us / Getting Help <contact>
   Contribution Guide <developer/contributing>
