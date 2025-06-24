Contributing to LSDB
===============================================================================

HATS and LSDB are primarily written and maintained by LINCC Frameworks, but we
would love to turn it over to the open-source scientific community!! We want to 
make sure that any discourse is open and inclusive, and we ask that everyone
involved read and adhere to the 
`LINCC Frameworks Code of Conduct <https://lsstdiscoveryalliance.org/programs/lincc-frameworks/code-conduct/>`_

Installation from Source
------------------------

To install the latest development version of LSDB you will want to build it from source. First, with your virtual environment activated, type in your terminal:

.. code-block:: bash

    git clone https://github.com/astronomy-commons/lsdb
    cd lsdb/

To install the package and dependencies you can run the ``setup_dev`` script which installs all 
the requirements to setup a development environment.

.. code-block:: bash

    chmod +x .setup_dev.sh
    ./.setup_dev.sh

Finally, to check that your package has been correctly installed, run the package unit tests:

.. code-block:: bash

    python -m pytest

Find (or make) a new GitHub issue
-------------------------------------------------------------------------------

Add yourself as the assignee on an existing issue so that we know who's working
on what. If you're not actively working on an issue, unassign yourself.

If there isn't an issue for the work you want to do, please create one and include
a description.

You can reach the team with bug reports, feature requests, and general inquiries
by creating a new GitHub issue.

Note that you may need to make changes in multiple repos to fully implement new
features or bug fixes! See related projects:

* HATS (`on GitHub <https://github.com/astronomy-commons/hats>`__ 
  and `on ReadTheDocs <https://hats.readthedocs.io/en/stable/>`__)
* HATS Import (`on GitHub <https://github.com/astronomy-commons/hats-import>`__
  and `on ReadTheDocs <https://hats-import.readthedocs.io/en/stable/>`__)

Fork the repository
-------------------------------------------------------------------------------

Contributing to LSDB requires you to `fork <https://github.com/astronomy-commons/lsdb/fork>`_ 
the GitHub repository. The next steps assume the creation of branches and PRs are performed from your fork.

.. note::
        
    If you are (or expect to be) a frequent contributor, you should consider requesting
    access to the `hats-friends <https://github.com/orgs/astronomy-commons/teams/hats-friends>`_
    working group. Members of this GitHub group should be able to create branches and PRs directly
    on LSDB, hats and hats-import, without the need of a fork.

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

We also have a handful of automated linters and checks using ``pre-commit``. You
can run against all staged changes with the command:

.. code-block:: bash

    pre-commit

Create a branch
-------------------------------------------------------------------------------

It is preferable that you create a new branch with a name like
``issue/##/<short-description>``. GitHub makes it pretty easy to associate
branches and tickets, but it's nice when it's in the name.

Create your PR
-------------------------------------------------------------------------------

You will be required to get your code approved before merging into main.
If you're not sure who to send it to, you can use the round-robin assignment
to the ``astronomy-commons/lincc-frameworks`` group.

We have a suite of continuous integration checks that run on PR creation. Please
follow the code quality recommendations of the linter and formatter, and make sure
every pipeline passes before submitting it for review.

Merge your PR
-------------------------------------------------------------------------------

When all the continuous integration checks have passed and upon receiving an
approving review, the author of the PR is welcome to merge it into the repository.

Release new version
-------------------------------------------------------------------------------

New versions are manually tagged and automatically released to pypi. To request
a new release of LSDB, HATS, and HATS import packages, create a 
`release ticket <https://github.com/astronomy-commons/lsdb/issues/new?assignees=delucchi-cmu&labels=&projects=&template=4-release_tracker.md&title=Release%3A+>`_

Contribute a tutorial notebook
-------------------------------------------------------------------------------

Tutorials follow general style guidelines; feel free to use our 
`tutorial notebook template <https://github.com/astronomy-commons/lsdb/blob/main/docs/developer/tutorial_template.ipynb>`__ 
to get started.
