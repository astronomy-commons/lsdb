==================
Input/Output
==================
.. currentmodule:: lsdb

Construction
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    open_catalog
    from_dataframe
    nested.datasets.generation.generate_catalog

Materializing
~~~~~~~~~~~~~~~~~~

Typically, you would materialize a catalog via ``catalog.write_catalog()`` calls,
and they will call these io methods with appropriate settings.

.. autosummary::
    :toctree: api/

    io.to_hats
    io.to_collection
    io.to_association
