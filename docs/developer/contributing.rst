Contributing to LSDB
===============================================================================

Find (or make) a new GitHub issue
-------------------------------------------------------------------------------

Add yourself as the assignee on an existing issue so that we know who's working
on what. If you're not actively working on an issue, unassign yourself.

If there isn't an issue for the work you want to do, please create one and include
a description.

You can reach the team with bug reports, feature requests, and general inquiries
by creating a new GitHub issue.

Fork the repository
-------------------------------------------------------------------------------

Contributing to LSDB requires you to `fork <https://github.com/astronomy-commons/lsdb/fork>`_ 
the GitHub repository. The next steps assume the creation of branches and PRs are performed from your fork.

.. note::
        
    If you are (or expect to be) a frequent contributor, you should consider requesting
    access to the `hipscat-friends <https://github.com/orgs/astronomy-commons/teams/hipscat-friends>`_
    working group. Members of this GitHub group should be able to create branches and PRs directly
    on LSDB, hipscat and hipscat-import, without the need of a fork.

Create a branch
-------------------------------------------------------------------------------

It is preferable that you create a new branch with a name like
``issue/##/<short-description>``. GitHub makes it pretty easy to associate
branches and tickets, but it's nice when it's in the name.

Set up a development environment
-------------------------------------------------------------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: bash

   >> conda create -n <env_name> python=3.10
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: bash

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/stable/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/stable/practices/sphinx.html#python-notebooks>`_.

.. tip::
    Installing on Mac

    Native prebuilt binaries for healpy on Apple Silicon Macs
    `do not yet exist <https://healpy.readthedocs.io/en/latest/install.html#binary-installation-with-pip-recommended-for-most-other-python-users>`_,
    so it's recommended to install via conda before proceeding to LSDB.

    .. code-block:: bash

        $ conda config --add channels conda-forge
        $ conda install healpy
        $ git clone https://github.com/astronomy-commons/lsdb
        $ cd lsdb
        $ pip install -e .

    When installing dev dependencies, make sure to include the single quotes.

    .. code-block:: bash

        $ pip install -e '.[dev]'

Testing
-------------------------------------------------------------------------------

Please add or update unit tests for all changes made to the codebase. You can run
unit tests locally simply with:

.. code-block:: bash

    pytest

If you're making changes to the sphinx documentation (anything under ``docs``),
you can build the documentation locally with a command like:

.. code-block:: bash

    cd docs
    make html

Create your PR
-------------------------------------------------------------------------------

Please use PR best practices, and get someone to review your code. We have a suite
of continuous integration tests that run on PR creation. Please follow the 
recommendations of the linter.

Merge your PR
-------------------------------------------------------------------------------

The author of the PR is welcome to merge their own PR into the repository.

Optional - Release a new version
-------------------------------------------------------------------------------

Once your PR is merged you can create a new release to make your changes available.
GitHub's `instructions <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>`_
for doing so are here.
Use your best judgement when incrementing the version. i.e. is this a major, minor, or patch fix.
