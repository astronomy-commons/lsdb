Tutorials
========================================================================================

These pages contain a set of tutorial notebooks for working through core and more advanced LSDB
functionality.

In general, you will find that most use cases follow a similar workflow that starts with
**LAZY** loading and planning operations, followed by more expensive **COMPUTE** operations.

.. graphviz::

   digraph workflow {
       compound=true;  nodesep=1.0;
        subgraph cluster0 {
            read_hipscat [label="read_hipscat, from_dataframe, ..." fontname="Courier" shape="none"];
            label="Read catalog structure";
        }

        subgraph cluster_plan_lazy {
            label="Plan computation"
            crossmatch [label="crossmatch, cone_search, ..." fontname="Courier" shape="none"]
        }
        subgraph cluster_compute {
            label="Read data / Do compute"
            compute [label="compute, to_hipscat, ..." fontname="Courier" shape="none"]
        }

        read_hipscat -> crossmatch  [ltail=cluster0, lhead=cluster_plan_lazy ]
        crossmatch -> compute  [ltail=cluster_plan_lazy, lhead=cluster_compute ]
   }

LSDB Introduction
---------------------------

An introduction to LSDB's core features and functionality

.. toctree::
    :maxdepth: 1

    Loading data into LSDB <tutorials/loading_data>
    Filtering large catalogs <tutorials/filtering_large_catalogs>
    Exporting results <tutorials/exporting_results>

Advanced Topics
---------------------------

A more in-depth look into how LSDB works

.. toctree::
    :maxdepth: 1
    :name: Advanced Topics

    Topic: Import catalogs <tutorials/import_catalogs>
    Topic: Margins <tutorials/margins>
    Topic: Performance Testing <tutorials/performance>

Example Science Use Cases
---------------------------

Notebooks going over some more scientific example use cases

.. toctree::
    :maxdepth: 1
    :name: Example Use Cases

    Science Notebook: Cross-match ZTF BTS and NGC <tutorials/pre_executed/ztf_bts-ngc>
    Science Notebook: Import and cross-match DES and Gaia <tutorials/pre_executed/des-gaia>
