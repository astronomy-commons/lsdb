VizieR Catalogs
========================================================================================

`VizieR <https://vizier.cds.unistra.fr/>`__ is an astronomical catalog service hosted by the Centre de Données astronomiques de Strasbourg (CDS), home to more than 25,000 catalogs. There are two ways to access VizieR catalogs with LSDB:

1. **VizieR HATS service** — for catalogs listed at `vizcat.cds.unistra.fr <https://vizcat.cds.unistra.fr/hats/>`__: load them directly with ``lsdb.open_catalog``.
2. **astroquery** — for any other VizieR catalog: fetch with ``astroquery`` and load into LSDB via ``lsdb.from_dataframe``.


1. VizieR HATS service
----------------------------------------------------------------------------------------

The `CDS VizieR HATS service <https://vizcat.cds.unistra.fr/hats/>`__ provides experimental HATS-on-the-fly access to a subset of VizieR catalogs. Catalogs available through this service can be loaded directly with ``lsdb.open_catalog``.

.. note::

   This service is in **beta**. Please keep in mind the following:

   - Only catalogs listed at `vizcat.cds.unistra.fr/hats <https://vizcat.cds.unistra.fr/hats/>`__ are accessible. If the catalog you need is not there, use the astroquery section below.
   - Do not use this service to retrieve full large catalogs such as 2MASS, AllWISE, or Gaia. For those, use :doc:`data.lsdb.io <datalsdb>` or convert the original catalog data with the `hats-import <https://hats-import.readthedocs.io/>`__ package.

The ``hats:n`` parameter in the URL controls the partition size:

- Use ``hats:n=10000`` when expecting to retrieve around 10,000 objects or fewer (e.g. a small cone search).
- Use ``hats:n=1000000`` (recommended default) for cross-matching and other large-scale tasks.


1.1. Example: load the Gaia DR3 AGN catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Gaia DR3 AGN catalog (I/358/vagn) <https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=I/358/vagn>`__ is available on the VizieR HATS service. Load it with LSDB:

.. code-block:: python

   import lsdb

   cat = lsdb.open_catalog("https://vizcat.cds.unistra.fr/hats:n=1000000/I/358/vagn/")

For a small cone search, use a smaller partition size to reduce data transfer:

.. code-block:: python

   import lsdb

   cat = lsdb.open_catalog("https://vizcat.cds.unistra.fr/hats:n=10000/I/358/vagn/")
   result = cat.cone_search(ra=187.0, dec=2.0, radius_arcsec=3600).compute()


2. Fetching from VizieR with astroquery
----------------------------------------------------------------------------------------

For catalogs not available on the VizieR HATS service, use `astroquery <https://astroquery.readthedocs.io/>`__ to fetch data and ``lsdb.from_astropy`` to load it into LSDB.

.. warning::

   This approach is only suitable for **small datasets** (fewer than ~1 million rows). For large VizieR catalogs, use :doc:`data.lsdb.io <datalsdb>` or convert the original data with the `hats-import <https://hats-import.readthedocs.io/>`__ package.


2.1. Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install astroquery

For more installation options, see the `astroquery documentation <https://astroquery.readthedocs.io/en/latest/#installation>`__.


2.2. Example: load the Gaia DR3 microturbulence catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Gaia DR3 microturbulence catalog (I/358/vmicro) <https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=I/358/vmicro>`__ is an example of a small catalog that can be fetched from VizieR with ``astroquery``:

.. code-block:: python

   import lsdb
   from astroquery.vizier import Vizier

   # Request _RAJ2000 and _DEJ2000 explicitly; "**" includes all native columns.
   # Most VizieR catalogs provide these computed columns (degrees, J2000),
   # and they are recommended as the ra/dec columns when creating HATS catalogs.
   vizier = Vizier(row_limit=1_000, columns=["_RAJ2000", "_DEJ2000", "**"])  # I/358/vmicro has 363 rows
   tables = vizier.get_catalogs("I/358/vmicro")

   cat = lsdb.from_astropy(tables[0], ra_column="_RAJ2000", dec_column="_DEJ2000", catalog_name="vmicro")
   cat.write_catalog("./vmicro")

We recommend saving the result as a HATS catalog immediately so you can reload it later without re-fetching:

.. code-block:: python

   cat = lsdb.open_catalog("./vmicro")

For a complete worked example including cross-matching, see the :doc:`ZTF BTS × NGC tutorial </tutorials/pre_executed/ztf_bts-ngc>`.
