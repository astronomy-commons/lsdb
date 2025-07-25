<img src="https://github.com/astronomy-commons/lsdb/blob/main/docs/lincc-logo.png?raw=true" width="300" height="100">


# LSDB

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/stable/)

[![PyPI](https://img.shields.io/pypi/v/lsdb?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/lsdb/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/lsdb.svg?color=blue&logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/lsdb) 

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/astronomy-commons/lsdb/smoke-test.yml)](https://github.com/astronomy-commons/lsdb/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/astronomy-commons/lsdb/branch/main/graph/badge.svg)](https://codecov.io/gh/astronomy-commons/lsdb)
[![Read the Docs](https://img.shields.io/readthedocs/lsdb)](https://lsdb.readthedocs.io/)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/astronomy-commons/lsdb/asv-main.yml?label=benchmarks)](https://astronomy-commons.github.io/lsdb/)

[![DOI:10.3847/2515-5172/ad4da1](https://zenodo.org/badge/DOI/10.48550/arXiv.2501.02103.svg)](https://ui.adsabs.harvard.edu/abs/2025arXiv250102103C)

## LSDB

LSDB is a python tool for scalable analysis of large catalogs 
(i.e. querying and crossmatching ~10⁹ sources). This package uses dask to parallelize operations across
multiple HATS partitioned surveys.

Check out our [ReadTheDocs site](https://docs.lsdb.io)
for more information on partitioning, installation, and contributing.

See related projects:

* HATS ([on GitHub](https://github.com/astronomy-commons/hats))
  ([on ReadTheDocs](https://hats.readthedocs.io/en/stable/))
* HATS Import ([on GitHub](https://github.com/astronomy-commons/hats-import))
  ([on ReadTheDocs](https://hats-import.readthedocs.io/en/stable/))

## Contributing

[![GitHub issue custom search in repo](https://img.shields.io/github/issues-search/astronomy-commons/lsdb?color=purple&label=Good%20first%20issues&query=is%3Aopen%20label%3A%22good%20first%20issue%22)](https://github.com/astronomy-commons/lsdb/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

See the [contribution guide](https://lsdb.readthedocs.io/en/stable/developer/contributing.html)
for complete installation instructions and contribution best practices.

## Citation

If you use LSDB in your work, please cite the conference proceedings: 
["Using LSDB to enable large-scale catalog distribution, cross-matching, and analytics"](https://ui.adsabs.harvard.edu/abs/2025arXiv250102103C). 

If you use Rubin Data Preview 1 (DP1) with LSDB, please also cite: ["Variability-finding in Rubin Data Preview 1 with LSDB"](https://ui.adsabs.harvard.edu/abs/2025arXiv250623955M).

Find full citation information [here](./CITATION.bib).

## Acknowledgements

This project is supported by Schmidt Sciences.

This project is based upon work supported by the National Science Foundation
under Grant No. AST-2003196.

This project acknowledges support from the DIRAC Institute in the Department of 
Astronomy at the University of Washington. The DIRAC Institute is supported 
through generous gifts from the Charles and Lisa Simonyi Fund for Arts and 
Sciences, and the Washington Research Foundation.
