Installation
============

The lastest release version of LSDB is available to install with `pip <https://pypi.org/project/lsdb/>`_ or `conda <https://anaconda.org/conda-forge/lsdb/>`_.

.. code-block:: bash

    conda install -c conda-forge lsdb

.. code-block:: bash

    python -m pip install lsdb

.. important::

    We recommend using a virtual environment. Before installing the package, create and activate a fresh environment called `lsdb_env` with conda, by typing in your terminal:

    .. code-block:: bash

        conda create -n lsdb_env python=3.11
        conda activate lsdb_env

    We recommend Python versions **>=3.9, <=3.12**.


Advanced Installation
---------------------

To install the latest development version of LSDB you will want to build it from source. First, with your virtual environment activated, type in your terminal:

.. code-block:: bash

    git clone https://github.com/astronomy-commons/lsdb
    cd lsdb/

To install the package and a minimum number of dependencies you can run:

.. code-block:: bash

    python -m pip install .
    python -m pip install pytest # to validate package installation

In alternative, you can execute the `setup_dev` script which installs all the additional requirements
to setup a development environment. Read more about contributing to LSDB in our :doc:`Contribution Guide <contributing>`.

.. code-block:: bash

    chmod +x .setup_dev.sh
    ./.setup_dev.sh

Finally, to check that your package has been correctly installed, run the package unit tests:

.. code-block:: bash

    python -m pytest


Creation of Jupyter Kernel
--------------------------

You may want to work with LSDB on Jupyter notebooks and, therefore, you need a kernel where
our package is installed. To install a kernel named `lsdb_kernel` for your environment, type:

.. code-block:: bash

    python -m ipykernel install --user --name lsdb_env --display-name "lsdb_kernel"

It should now be available for selection in your Jupyter dashboard!
