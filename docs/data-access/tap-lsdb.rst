TAP Access via tap.data.lsdb.io
========================================================================================

`tap.data.lsdb.io <https://tap.data.lsdb.io>`__ is an experimental implementation of
the `Table Access Protocol (TAP) <https://www.ivoa.net/documents/TAP/>`__, an IVOA
standard for querying astronomical catalogs using ADQL (Astronomical Data Query Language).

.. warning::

   This service is **experimental**. Not all ADQL features are supported, and behavior
   may change. If you run into problems, please `open an issue
   <https://github.com/astronomy-commons/hats-tap/issues>`__ or contact the LINCC
   Frameworks team.

How It Works
----------------------------------------------------------------------------------------

Queries submitted to ``tap.data.lsdb.io`` are handled by
`hats-tap <https://github.com/astronomy-commons/hats-tap>`__, an open-source Python
package that acts as a TAP-compatible server for HATS/LSDB catalogs.

When a query arrives, hats-tap:

1. **Parses** the ADQL query using `queryparser <https://pypi.org/project/queryparser/>`__.
2. **Translates** it into an equivalent sequence of `LSDB <https://lsdb.readthedocs.io>`__
   Python calls — for example, mapping a cone search to ``lsdb.ConeSearch`` and column
   filters to ``catalog.query()``.
3. **Executes** the translated query against the HATS catalogs hosted at
   `data.lsdb.io <https://data.lsdb.io>`__.
4. **Returns** results in VOTable format, compatible with tools like
   `TOPCAT <https://www.star.bris.ac.uk/~mbt/topcat/>`__ and
   `PyVO <https://pyvo.readthedocs.io/>`__.

Endpoints
----------------------------------------------------------------------------------------

The service exposes the following endpoints:

- **GET/POST** ``/sync`` — submit a synchronous ADQL query
- **GET** ``/tables`` — list available catalogs and their columns
- **GET** ``/capabilities`` — service metadata following the TAP standard

Example Query
----------------------------------------------------------------------------------------

Submit a query using ``curl``:

.. code-block:: bash

   curl -X POST https://tap.data.lsdb.io/sync \
     -d "REQUEST=doQuery" \
     -d "LANG=ADQL" \
     -d "QUERY=SELECT TOP 10 ra, dec, mean_mag_g, mean_mag_r, mean_mag_i \
                  FROM ztf_dr14 \
                  WHERE mean_mag_r < 20"

Or from Python using PyVO:

.. code-block:: python

   import pyvo
   service = pyvo.dal.TAPService("https://tap.data.lsdb.io")
   results = service.search("""
       SELECT TOP 15
           source_id, ra, dec, phot_g_mean_mag
       FROM gaia_dr3.gaia
       WHERE 1 = CONTAINS(
           POINT('ICRS', ra, dec),
           CIRCLE('ICRS', 270.0, 23.0, 0.25)
       )
       AND phot_g_mean_mag < 16
   """)
   results.to_table()

Supported ADQL Features
----------------------------------------------------------------------------------------

The following ADQL constructs are currently translated and executed:

**Column selection**
   Explicit column lists in the ``SELECT`` clause are supported.

   .. code-block:: sql

      SELECT ra, dec, phot_g_mean_mag FROM gaia_dr3.gaia

**Result limiting**
   The ``TOP`` keyword limits the number of returned rows.

   .. code-block:: sql

      SELECT TOP 100 ra, dec FROM ztf_dr14

**Row filtering**
   ``WHERE`` clauses with comparison operators (``<``, ``>``, ``<=``, ``>=``, ``=``)
   and multiple conditions combined with ``AND`` are supported.

   .. code-block:: sql

      SELECT ra, dec FROM ztf_dr14 WHERE mean_mag_r < 20 AND nepochs > 5

**Cone search**
   Spatial filtering using ``CONTAINS`` with ``POINT`` and ``CIRCLE`` is translated to
   ``lsdb.ConeSearch``. Only one ``CONTAINS`` clause per query is supported.

   .. code-block:: sql

      SELECT ra, dec FROM gaia_dr3.gaia
      WHERE 1 = CONTAINS(
          POINT('ICRS', ra, dec),
          CIRCLE('ICRS', 270.0, 23.0, 0.25)
      )

**Polygon search**
   ``CONTAINS`` with ``POINT`` and ``POLYGON`` is translated to ``lsdb.PolygonSearch``.

   .. code-block:: sql

      SELECT ra, dec FROM ztf_dr22
      WHERE CONTAINS(
          POINT('ICRS', ra, dec),
          POLYGON('ICRS', 280.0, 30.0, 281.0, 30.0, 281.0, 29.0, 279.0, 27.0)
      ) = 1

**Ordering**
   ``ORDER BY`` is supported for sorting results.

   .. code-block:: sql

      SELECT ra, dec FROM ztf_dr22
      WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 280.0, 0.0, 0.1)) = 1
      ORDER BY ra, dec DESC

Not Yet Supported
----------------------------------------------------------------------------------------

The following ADQL features are recognized by the standard but are not yet implemented.
Queries using them will return an error.

- **SELECT \*** — wildcard column selection is not supported; column names must be
  listed explicitly (`issue #22 <https://github.com/astronomy-commons/hats-tap/issues/22>`__).
- **BETWEEN** — range predicates (``col BETWEEN a AND b``) are not yet translated
  (`issue #17 <https://github.com/astronomy-commons/hats-tap/issues/17>`__).
- **Arithmetic expressions** — computed columns or filter values involving arithmetic
  (e.g. ``ra + 10``) are not supported
  (`issue #10 <https://github.com/astronomy-commons/hats-tap/issues/10>`__).
- **Aggregation functions** — ``COUNT``, ``MIN``, ``MAX``, ``SUM``, ``AVG``, and
  similar aggregate functions are not yet implemented
  (`issue #20 <https://github.com/astronomy-commons/hats-tap/issues/20>`__).
- **COUNT(DISTINCT ...)** — unique-value counting is not supported
  (`issue #19 <https://github.com/astronomy-commons/hats-tap/issues/19>`__).
- **Multiple CONTAINS clauses** — only a single spatial constraint per query is
  allowed.
- **POINT or CIRCLE outside of CONTAINS** — these geometry functions must appear
  inside a ``CONTAINS`` expression.
- **OR conditions** — only ``AND``-connected predicates in the ``WHERE`` clause are
  supported.
- **JOIN** — multi-table joins are not yet translated
  (`issue #16 <https://github.com/astronomy-commons/hats-tap/issues/16>`__).
- **Subqueries** — nested ``SELECT`` statements are not supported.
- **ID lookups (``id_search``)** — queries that filter on a catalog's primary ID column
  are not yet optimized via ``lsdb.id_search``
  (`issue #14 <https://github.com/astronomy-commons/hats-tap/issues/14>`__).
- **Nearest-neighbor / self-join** — crossmatch-style queries are not yet supported
  (`issue #15 <https://github.com/astronomy-commons/hats-tap/issues/15>`__).

Getting Help
----------------------------------------------------------------------------------------

If a query fails or you encounter unexpected behavior, please
`open an issue on GitHub <https://github.com/astronomy-commons/hats-tap/issues>`__
with the query you submitted and any error message you received.
