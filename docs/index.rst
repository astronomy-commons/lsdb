.. lsdb documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LSDB - Large Survey DataBase
========================================================================================

LSDB is a framework that facilitates and enables fast spatial analysis for extremely large astronomical catalogs
(i.e. querying and crossmatching O(1B) sources). It aims to address large-scale data processing challenges, in
particular those brought up by `LSST <https://www.lsst.org/about>`_.

Built on top of Dask to efficiently scale and parallelize operations across multiple workers, it leverages
the `HiPSCat <https://hipscat.readthedocs.io/en/latest/>`_ data format for surveys in a partitioned HEALPix
(Hierarchical Equal Area isoLatitude Pixelization) structure.

In this website you will find:

- :doc:`Getting Started <gettingstarted>` guides on how to install and run an example workflow
- :doc:`Tutorials <tutorials>`, and :doc:`Notebooks <notebooks>` with more advanced usage examples
- The detailed :doc:`API Reference <autoapi/index>` documentation

Learn more about contributing to this repository in our :doc:`Contribution Guide <gettingstarted/contributing>`.

.. toctree::
   :hidden:

   Home page <self>
   Getting Started <gettingstarted>
   Tutorials <tutorials>
   Notebooks <notebooks>
   API Reference <autoapi/index>

