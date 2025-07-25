.. lsdb documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


LSDB
========================================================================================

LSDB is a python tool for scalable analysis of large catalogs (e.g. querying 
and crossmatching ~10⁹ sources). It aims to address large-scale data processing 
challenges, in particular those brought up by `LSST <https://www.lsst.org/about>`_.

Built on top of `Dask <https://docs.dask.org/>`_ to efficiently scale and parallelize operations across multiple distributed workers, it
uses the `HATS <https://hats.readthedocs.io/en/stable/>`_ data format to efficiently perform spatial
operations.

.. figure:: _static/gaia.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: A possible HEALPix distribution for Gaia DR3

   A possible HEALPix distribution for Gaia DR3.


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

.. grid:: 1 1 2 2

   .. grid-item-card:: API Reference
       :link: reference
       :link-type: doc

       The detailed API documentation

   .. grid-item-card:: Contact us
       :link: contact
       :link-type: doc

       Reach out for more help

.. toctree::
   :hidden:
   :caption: Using LSDB

   Home page <self>
   Getting Started <getting-started>
   Tutorials <tutorials>
   API Reference <reference>

.. toctree::
   :hidden:
   :caption: Project

   About & Citation <citation>
   Getting Help <contact>
   Contribution Guide <developer/contributing>
