.. lsdb documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LSDB - Large Survey DataBase
========================================================================================

LSDB is a framework that facilitates and enables spatial analysis for extremely large astronomical catalogs
(i.e. querying and crossmatching O(1B) sources). It is built on top of Dask to parallelize operations across
multiple surveys partitioned with an `HiPSCat <https://hipscat.readthedocs.io/en/latest/>`_ structure.

In this website you will find tutorials on how to get started with LSDB, as well as its API reference. If you
wish to contribute to the repository please visit our :doc:`Contribution Guide <gettingstarted/contributing>`.

.. toctree::
   :hidden:

   Home page <self>
   Getting Started <gettingstarted>
   Tutorials <tutorials>
   Notebooks <notebooks>
   API Reference <autoapi/index>

