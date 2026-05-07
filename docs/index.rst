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

    <div class="api-surface-callout" role="note">
       Interactive map: hover to magnify and click any labeled box to open its API reference page.
    </div>
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
                      data-magnify-zoom="2.0"
                     data-magnify-size="300"
               class="api-surface-image" />
                   <div class="api-surface-magnifier" aria-hidden="true"></div>
       <map name="lsdb-api-surface-map">
          <area shape="rect" coords="126,178,549,243" href="reference/api/lsdb.open_catalog.html" alt="open_catalog" />
          <area shape="rect" coords="126,238,549,303" href="reference/api/lsdb.from_dataframe.html" alt="from_dataframe" />
          <area shape="rect" coords="126,298,549,364" href="reference/api/lsdb.from_astropy.html" alt="from_astropy" />
          <area shape="rect" coords="126,358,549,424" href="reference/api/lsdb.nested.datasets.generation.generate_catalog.html" alt="generate_catalog" />
          <area shape="rect" coords="417,632,721,690" href="reference/api/lsdb.show_versions.html" alt="show_versions" />
          <area shape="rect" coords="487,802,829,868" href="reference/api/lsdb.catalog.Catalog.plot_pixels.html" alt="plot_pixels" />
          <area shape="rect" coords="487,862,829,928" href="reference/api/lsdb.catalog.Catalog.plot_coverage.html" alt="plot_coverage" />
          <area shape="rect" coords="498,967,782,1033" href="reference/api/lsdb.catalog.Catalog.plot_points.html" alt="plot_points" />

          <area shape="rect" coords="359,1217,555,1274" href="reference/api/lsdb.catalog.Catalog.html" alt="Catalog" />
          <area shape="rect" coords="1010,306,1576,372" href="reference/catalog_properties.html" alt="name" />
          <area shape="rect" coords="1010,366,1218,432" href="reference/api/lsdb.catalog.Catalog.columns.html" alt="columns" />
          <area shape="rect" coords="1220,366,1539,432" href="reference/api/lsdb.catalog.Catalog.all_columns.html" alt="all_columns" />
          <area shape="rect" coords="1010,426,1539,492" href="reference/api/lsdb.catalog.Catalog.nested_columns.html" alt="nested_columns" />
          <area shape="rect" coords="1010,486,1384,552" href="reference/api/lsdb.catalog.Catalog.original_schema.html" alt="original_schema" />
          <area shape="rect" coords="1386,486,1546,552" href="reference/api/lsdb.catalog.Catalog.dtypes.html" alt="dtypes" />
          <area shape="rect" coords="1010,590,1160,654" href="reference/api/lsdb.catalog.Catalog.head.html" alt="head" />
          <area shape="rect" coords="1162,590,1376,654" href="reference/api/lsdb.catalog.Catalog.tail.html" alt="tail" />
          <area shape="rect" coords="1010,646,1181,709" href="reference/api/lsdb.catalog.Catalog.sample.html" alt="sample" />
          <area shape="rect" coords="1183,646,1535,709" href="reference/api/lsdb.catalog.Catalog.random_sample.html" alt="random_sample" />
          <area shape="rect" coords="1016,740,1188,806" href="reference/api/lsdb.catalog.Catalog.rename.html" alt="rename" />
          <area shape="rect" coords="1826,269,2147,335" href="reference/api/lsdb.catalog.Catalog.hc_structure.html" alt="hc_structure" />
          <area shape="rect" coords="1817,399,2135,465" href="reference/catalog_properties.html" alt="hc_collection" />
          <area shape="rect" coords="1764,551,2207,655" href="reference/api/lsdb.catalog.MarginCatalog.html" alt="MarginCatalog" />
          <area shape="rect" coords="1813,653,2161,744" href="reference/api/lsdb.catalog.MapCatalog.html" alt="MapCatalog" />
         <area shape="rect" coords="1757,792,2218,858" href="reference/api/lsdb.io.to_association.html" alt="AssociationCatalog" />

          <area shape="rect" coords="1233,948,1308,1014" href="reference/api/lsdb.catalog.Catalog.__len__.html" alt="len" />
          <area shape="rect" coords="974,1044,1664,1110" href="reference/api/lsdb.catalog.Catalog.aggregate_column_statistics.html" alt="aggregate_column_statistics" />
          <area shape="rect" coords="1666,1044,2181,1110" href="reference/api/lsdb.catalog.Catalog.per_pixel_statistics.html" alt="per_pixel_statistics" />
          <area shape="rect" coords="974,1148,1451,1213" href="reference/api/lsdb.catalog.Catalog.get_healpix_pixels.html" alt="get_healpix_pixels" />
          <area shape="rect" coords="1453,1148,2136,1213" href="reference/api/lsdb.catalog.Catalog.get_ordered_healpix_pixels.html" alt="get_ordered_healpix_pixels" />
          <area shape="rect" coords="974,1210,1228,1275" href="reference/api/lsdb.catalog.Catalog.partitions.html" alt="partitions" />
          <area shape="rect" coords="1230,1210,1541,1275" href="reference/api/lsdb.catalog.Catalog.npartitions.html" alt="npartitions" />
          <area shape="rect" coords="1543,1210,2068,1275" href="reference/api/lsdb.catalog.Catalog.get_partition_index.html" alt="get_partition_index" />
          <area shape="rect" coords="981,1319,1333,1384" href="reference/api/lsdb.catalog.Catalog.estimate_size.html" alt="estimate_size" />
          <area shape="rect" coords="1046,1836,1398,1902" href="reference/api/lsdb.catalog.Catalog.get_partition.html" alt="get_partition" />

          <area shape="rect" coords="1022,1593,1593,1658" href="reference/api/lsdb.catalog.Catalog.prune_empty_partitions.html" alt="prune_empty_partitions" />
          <area shape="rect" coords="1651,1486,2108,1567" href="reference/api/lsdb.catalog.Catalog.map_partitions.html" alt="map_partitions" />
          <area shape="rect" coords="1651,1572,2108,1653" href="reference/api/lsdb.catalog.Catalog.map_rows.html" alt="map_rows" />
          <area shape="rect" coords="1046,1697,1556,1763" href="reference/api/lsdb.catalog.Catalog.to_dask_dataframe.html" alt="to_dask_dataframe" />
          <area shape="rect" coords="1046,1766,1556,1832" href="reference/api/lsdb.catalog.Catalog.to_delayed.html" alt="to_delayed" />

          <area shape="rect" coords="2510,342,2650,408" href="reference/api/lsdb.catalog.Catalog.query.html" alt="query" />
          <area shape="rect" coords="2497,454,2834,514" href="reference/api/lsdb.catalog.Catalog.box_search.html" alt="box_search" />
          <area shape="rect" coords="2497,509,2834,569" href="reference/api/lsdb.catalog.Catalog.cone_search.html" alt="cone_search" />
          <area shape="rect" coords="2497,563,2834,623" href="reference/api/lsdb.catalog.Catalog.polygon_search.html" alt="polygon_search" />
          <area shape="rect" coords="2492,665,2815,730" href="reference/api/lsdb.catalog.Catalog.order_search.html" alt="order_search" />
          <area shape="rect" coords="2492,725,2815,791" href="reference/api/lsdb.catalog.Catalog.pixel_search.html" alt="pixel_search" />
          <area shape="rect" coords="2508,823,2748,889" href="reference/api/lsdb.catalog.Catalog.id_search.html" alt="id_search" />
          <area shape="rect" coords="2507,917,2676,983" href="reference/api/lsdb.catalog.Catalog.search.html" alt="search" />
          <area shape="rect" coords="2504,1019,2788,1085" href="reference/api/lsdb.catalog.Catalog.moc_search.html" alt="moc_search" />

          <area shape="rect" coords="3106,372,3576,438" href="reference/api/lsdb.catalog.Catalog.crossmatch.html" alt="crossmatch" />
          <area shape="rect" coords="3106,432,3576,498" href="reference/api/lsdb.catalog.Catalog.crossmatch_nested.html" alt="crossmatch_nested" />
          <area shape="rect" coords="3120,693,3331,758" href="reference/api/lsdb.catalog.Catalog.merge.html" alt="merge" />
          <area shape="rect" coords="3120,753,3409,819" href="reference/api/lsdb.catalog.Catalog.concat.html" alt="concat" />
          <area shape="rect" coords="3120,873,3409,939" href="reference/api/lsdb.catalog.Catalog.join.html" alt="join" />
          <area shape="rect" coords="3120,933,3409,999" href="reference/api/lsdb.catalog.Catalog.join_nested.html" alt="join_nested" />
          <area shape="rect" coords="3120,1054,3409,1119" href="reference/api/lsdb.catalog.Catalog.nest_lists.html" alt="nest_lists" />
          <area shape="rect" coords="3266,1247,3549,1313" href="reference/api/lsdb.catalog.Catalog.merge_asof.html" alt="merge_asof" />
          <area shape="rect" coords="3266,1307,3549,1373" href="reference/api/lsdb.catalog.Catalog.merge_map.html" alt="merge_map" />

          <area shape="rect" coords="2513,1346,2832,1405" href="reference/api/lsdb.streams.CatalogStream.html" alt="CatalogStream" />
          <area shape="rect" coords="2513,1399,2832,1459" href="reference/api/lsdb.streams.InfiniteStream.html" alt="InfiniteStream" />

          <area shape="rect" coords="1761,1906,2109,2015" href="reference/api/lsdb.catalog.Catalog.compute.html" alt="compute" />
          <area shape="rect" coords="2244,1906,2808,2010" href="reference/api/lsdb.catalog.Catalog.write_catalog.html" alt="write_catalog" />
          <area shape="rect" coords="2902,1822,3113,1888" href="reference/api/lsdb.io.to_hats.html" alt="to_hats" />
          <area shape="rect" coords="3177,1824,3546,1890" href="reference/api/lsdb.io.to_association.html" alt="to_association" />
       </map>
    </div>

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
