Installation
============

The lastest release version of LSDB is available to install with `pip <https://pypi.org/project/lsdb/>`_ or `conda <https://anaconda.org/conda-forge/lsdb/>`_.

.. code-block:: bash

    python -m pip install lsdb

.. code-block:: bash

    conda install -c conda-forge lsdb

.. important::

    We recommend using a virtual environment. Before installing the package, create and activate a fresh environment called `lsdb_env` with conda, by typing in your terminal:

    .. code-block:: bash

        conda create -n lsdb_env python=3.11
        conda activate lsdb_env

    We recommend Python versions **>=3.9, <=3.12**.

LSDB can also be installed from the source on `GitHub <https://github.com/astronomy-commons/lsdb>`. See our
advanced installation instructions in the :doc:`contributing guide <developer/contributing>`

Creation of Jupyter Kernel
--------------------------

You may want to work with LSDB on Jupyter notebooks and, therefore, you need a kernel where
our package is installed. To install a kernel named `lsdb_kernel` for your environment, type:

.. code-block:: bash

    python -m ipykernel install --user --name lsdb_env --display-name "lsdb_kernel"

It should now be available for selection in your Jupyter dashboard!
