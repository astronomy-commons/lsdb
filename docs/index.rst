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
               class="api-surface-image" />
       <map name="lsdb-api-surface-map">
          <area shape="rect" coords="29,91,233,186" href="reference/api/lsdb.open_catalog.html" alt="open_catalog" />
          <area shape="rect" coords="29,118,252,215" href="reference/api/lsdb.from_dataframe.html" alt="from_dataframe" />
          <area shape="rect" coords="29,143,236,238" href="reference/api/lsdb.from_astropy.html" alt="from_astropy" />
          <area shape="rect" coords="29,167,271,259" href="reference/api/lsdb.nested.datasets.generation.generate_catalog.html" alt="generate_catalog" />
          <area shape="rect" coords="26,243,240,339" href="reference/api/lsdb.show_versions.html" alt="show_versions" />
          <area shape="rect" coords="26,328,241,439" href="reference/api/lsdb.catalog.Catalog.plot_pixels.html" alt="plot_pixels" />
          <area shape="rect" coords="26,350,258,459" href="reference/api/lsdb.catalog.Catalog.plot_coverage.html" alt="plot_coverage" />
          <area shape="rect" coords="26,374,229,483" href="reference/api/lsdb.catalog.Catalog.plot_points.html" alt="plot_points" />

          <area shape="rect" coords="360,130,489,214" href="reference/api/lsdb.catalog.Catalog.html" alt="Catalog" />
          <area shape="rect" coords="356,161,448,253" href="reference/catalog_properties.html" alt="name" />
          <area shape="rect" coords="355,184,544,279" href="reference/api/lsdb.catalog.Catalog.columns.html" alt="columns" />
          <area shape="rect" coords="355,206,612,302" href="reference/api/lsdb.catalog.Catalog.all_columns.html" alt="all_columns" />
          <area shape="rect" coords="355,227,568,323" href="reference/api/lsdb.catalog.Catalog.nested_columns.html" alt="nested_columns" />
          <area shape="rect" coords="355,250,617,344" href="reference/api/lsdb.catalog.Catalog.original_schema.html" alt="original_schema" />
          <area shape="rect" coords="355,272,468,364" href="reference/api/lsdb.catalog.Catalog.dtypes.html" alt="dtypes" />
          <area shape="rect" coords="355,301,432,393" href="reference/api/lsdb.catalog.Catalog.head.html" alt="head" />
          <area shape="rect" coords="435,301,503,393" href="reference/api/lsdb.catalog.Catalog.tail.html" alt="tail" />
          <area shape="rect" coords="355,326,450,417" href="reference/api/lsdb.catalog.Catalog.sample.html" alt="sample" />
          <area shape="rect" coords="451,326,652,417" href="reference/api/lsdb.catalog.Catalog.random_sample.html" alt="random_sample" />
          <area shape="rect" coords="355,353,453,447" href="reference/api/lsdb.catalog.Catalog.rename.html" alt="rename" />
          <area shape="rect" coords="642,125,804,212" href="reference/api/lsdb.catalog.Catalog.hc_structure.html" alt="hc_structure" />
          <area shape="rect" coords="642,172,802,258" href="reference/catalog_properties.html" alt="hc_collection" />
          <area shape="rect" coords="629,214,812,312" href="reference/api/lsdb.catalog.MarginCatalog.html" alt="MarginCatalog" />
          <area shape="rect" coords="629,259,782,353" href="reference/api/lsdb.catalog.MapCatalog.html" alt="MapCatalog" />
         <area shape="rect" coords="629,301,866,393" href="reference/api/lsdb.io.to_association.html" alt="AssociationCatalog" />

          <area shape="rect" coords="337,432,410,527" href="reference/api/lsdb.catalog.Catalog.__len__.html" alt="len" />
          <area shape="rect" coords="304,460,626,552" href="reference/api/lsdb.catalog.Catalog.aggregate_column_statistics.html" alt="aggregate_column_statistics" />
          <area shape="rect" coords="629,460,887,552" href="reference/api/lsdb.catalog.Catalog.per_pixel_statistics.html" alt="per_pixel_statistics" />
          <area shape="rect" coords="304,507,540,603" href="reference/api/lsdb.catalog.Catalog.get_healpix_pixels.html" alt="get_healpix_pixels" />
          <area shape="rect" coords="540,507,853,603" href="reference/api/lsdb.catalog.Catalog.get_ordered_healpix_pixels.html" alt="get_ordered_healpix_pixels" />
          <area shape="rect" coords="853,507,1010,603" href="reference/api/lsdb.catalog.Catalog.partitions.html" alt="partitions" />
          <area shape="rect" coords="304,533,467,628" href="reference/api/lsdb.catalog.Catalog.npartitions.html" alt="npartitions" />
          <area shape="rect" coords="466,533,667,628" href="reference/api/lsdb.catalog.Catalog.get_partition_index.html" alt="get_partition_index" />
          <area shape="rect" coords="304,575,463,668" href="reference/api/lsdb.catalog.Catalog.estimate_size.html" alt="estimate_size" />
          <area shape="rect" coords="465,575,615,668" href="reference/api/lsdb.catalog.Catalog.get_partition.html" alt="get_partition" />

          <area shape="rect" coords="321,646,574,741" href="reference/api/lsdb.catalog.Catalog.prune_empty_partitions.html" alt="prune_empty_partitions" />
          <area shape="rect" coords="578,620,776,707" href="reference/api/lsdb.catalog.Catalog.map_partitions.html" alt="map_partitions" />
          <area shape="rect" coords="580,668,730,751" href="reference/api/lsdb.catalog.Catalog.map_rows.html" alt="map_rows" />
          <area shape="rect" coords="363,683,596,767" href="reference/api/lsdb.catalog.Catalog.to_dask_dataframe.html" alt="to_dask_dataframe" />
          <area shape="rect" coords="363,711,537,797" href="reference/api/lsdb.catalog.Catalog.to_delayed.html" alt="to_delayed" />
          <area shape="rect" coords="362,735,512,820" href="reference/api/lsdb.catalog.Catalog.get_partition.html" alt="get_partition execution" />

          <area shape="rect" coords="937,64,1101,157" href="reference/api/lsdb.catalog.Catalog.query.html" alt="query" />
          <area shape="rect" coords="937,115,1136,300" href="reference/api/lsdb.catalog.Catalog.box_search.html" alt="box_search" />
          <area shape="rect" coords="937,136,1142,323" href="reference/api/lsdb.catalog.Catalog.cone_search.html" alt="cone_search" />
          <area shape="rect" coords="937,158,1170,347" href="reference/api/lsdb.catalog.Catalog.polygon_search.html" alt="polygon_search" />
          <area shape="rect" coords="937,208,1137,395" href="reference/api/lsdb.catalog.Catalog.order_search.html" alt="order_search" />
          <area shape="rect" coords="937,233,1137,420" href="reference/api/lsdb.catalog.Catalog.pixel_search.html" alt="pixel_search" />
          <area shape="rect" coords="937,263,1120,444" href="reference/api/lsdb.catalog.Catalog.id_search.html" alt="id_search" />
          <area shape="rect" coords="937,288,1072,468" href="reference/api/lsdb.catalog.Catalog.search.html" alt="search" />
          <area shape="rect" coords="937,317,1164,495" href="reference/api/lsdb.catalog.Catalog.moc_search.html" alt="moc_search" />

          <area shape="rect" coords="1095,97,1276,191" href="reference/api/lsdb.catalog.Catalog.crossmatch.html" alt="crossmatch" />
          <area shape="rect" coords="1095,121,1300,216" href="reference/api/lsdb.catalog.Catalog.crossmatch_nested.html" alt="crossmatch_nested" />
          <area shape="rect" coords="1097,214,1279,468" href="reference/api/lsdb.catalog.Catalog.merge.html" alt="merge" />
          <area shape="rect" coords="1097,238,1223,492" href="reference/api/lsdb.catalog.Catalog.concat.html" alt="concat" />
          <area shape="rect" coords="1097,293,1179,549" href="reference/api/lsdb.catalog.Catalog.join.html" alt="join" />
          <area shape="rect" coords="1097,319,1270,573" href="reference/api/lsdb.catalog.Catalog.join_nested.html" alt="join_nested" />
          <area shape="rect" coords="1097,346,1243,600" href="reference/api/lsdb.catalog.Catalog.nest_lists.html" alt="nest_lists" />
          <area shape="rect" coords="1194,437,1323,532" href="reference/api/lsdb.catalog.Catalog.merge_asof.html" alt="merge_asof" />
          <area shape="rect" coords="1194,462,1319,556" href="reference/api/lsdb.catalog.Catalog.merge_map.html" alt="merge_map" />

          <area shape="rect" coords="943,501,1096,597" href="reference/api/lsdb.streams.CatalogStream.html" alt="CatalogStream" />
          <area shape="rect" coords="943,527,1096,620" href="reference/api/lsdb.streams.InfiniteStream.html" alt="InfiniteStream" />

          <area shape="rect" coords="768,730,893,812" href="reference/api/lsdb.catalog.Catalog.compute.html" alt="compute" />
          <area shape="rect" coords="900,730,1110,812" href="reference/api/lsdb.catalog.Catalog.write_catalog.html" alt="write_catalog" />
          <area shape="rect" coords="1177,739,1283,822" href="reference/api/lsdb.io.to_hats.html" alt="to_hats" />
          <area shape="rect" coords="1295,739,1441,822" href="reference/api/lsdb.io.to_association.html" alt="to_association" />
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
