.. lsdb documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


LSDB
========================================================================================

LSDB (Large Survey DataBase) is a python framework that enables simple, fast spatial analysis of extremely
large astronomical catalogs (e.g. querying and crossmatching O(1B) sources). It aims to address large-scale
data processing challenges, in particular those brought up by `LSST <https://www.lsst.org/about>`_.

Built on top of Dask to efficiently scale and parallelize operations across multiple distributed workers, it
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
       :link: autoapi/index
       :link-type: doc

       The detailed API documentation

   .. grid-item-card:: Contribution Guide
       :link: developer/contributing
       :link-type: doc

       For developers, learn more about contributing to this repository

.. toctree::
   :hidden:

   Home page <self>
   Getting Started <getting-started>
   Tutorials <tutorials>
   API Reference <autoapi/index>

.. toctree::
   :hidden:

   Contact us <contact>

.. toctree::
   :hidden:
   :caption: Developers

   Contribution Guide <developer/contributing>

Acknowledgements
-------------------------------------------------------------------------------

This project is supported by Schmidt Sciences.

This project is based upon work supported by the National Science Foundation
under Grant No. AST-2003196.

This project acknowledges support from the DIRAC Institute in the Department of 
Astronomy at the University of Washington. The DIRAC Institute is supported 
through generous gifts from the Charles and Lisa Simonyi Fund for Arts and 
Sciences, and the Washington Research Foundation.