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

Core functionality
------------------
.. toctree::
    :maxdepth: 1

    Loading data into LSDB <tutorials/loading_data>
    Working with large catalogs <tutorials/working_with_large_catalogs>
    Margins <tutorials/margins>
    Exporting results <tutorials/exporting_results>

Other tutorials
---------------
.. toctree::
    :maxdepth: 1

    Import catalogs <tutorials/import_catalogs>
    Cross-match ZTF BTS and NGC <tutorials/pre_executed/ztf_bts-ngc>
    Import and cross-match DES and Gaia <tutorials/pre_executed/des-gaia>
