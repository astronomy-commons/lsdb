Installation
============

The lastest release version of LSDB is available to install with `pip <https://pypi.org/project/lsdb/>`_ or `conda <https://anaconda.org/conda-forge/lsdb/>`_.

.. code-block:: bash

    pip install lsdb

.. code-block:: bash

    conda install -c conda-forge lsdb

.. important::

    We recommend installing the package inside a virtual environment.
    
    To create a fresh environment called `lsdb_env` with conda, type:

    .. code-block:: bash

        conda create -n lsdb_env python=3.10
        conda activate lsdb_env

    We recommend Python versions >=3.9, <=3.12.


Advanced Installation
---------------------

If you're looking to install the latest development version of LSDB you will want to build it from source.

With your virtual environment activated, type in your terminal:

.. code-block:: bash

    git clone https://github.com/astronomy-commons/lsdb
    cd lsdb/
    ./setup_dev.sh

The `setup_dev` script installs additional requirements to setup a development environment,
enabling us to run unit tests and build documentation. 

To check that your package has been correctly installed, run the package unit tests:

.. code-block:: bash

    pytest


Creation of Jupyter Kernel
--------------------------

You may want to create Jupyter notebooks to work with LSDB and therefore you need a kernel.

To install a kernel named `lsdb_kernel`  for your environment, type:

.. code-block:: bash

    python -m ipykernel install --user --name lsdb_env --display-name "lsdb_kernel"

It should now be available for selection in your Jupyter dashboard!
