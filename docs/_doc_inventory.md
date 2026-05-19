# LSDB Documentation Inventory

> Temporary working document for docs restructuring. Delete when reorg is complete.
>
> Audience context: target reader is an experienced astronomer (PhD-level) who is new to LSDB.
> Difficulty ratings below are relative to that baseline.

---

## Contents

1. [Prose / RST docs](#prose--rst-docs)
2. [Core tutorial notebooks](#core-tutorial-notebooks)
3. [Pre-executed: technique notebooks](#pre-executed-technique-notebooks)
4. [Pre-executed: science example notebooks](#pre-executed-science-example-notebooks)
5. [Pre-executed: Rubin-specific notebooks](#pre-executed-rubin-specific-notebooks)
6. [Cross-cutting observations](#cross-cutting-observations)

---

## Prose / RST docs

### `index.rst`
**Title:** LSDB (landing page)  
**Role:** Entry point and navigation hub.  
LSDB is a Python tool for scalable analysis of large astronomical catalogs (~10⁹ sources), built on Dask and the HATS format. Contains an interactive API surface image map for navigation.
- **Notable:** Image map (`API_Surface_Feb_12.png`) may not render well in all viewers.

---

### `getting-started.rst`
**Title:** Getting Started with LSDB  
**Role:** Quick-start guide for new users.  
Covers environment setup (conda/venv/pyenv), pip/conda installation, and a minimal workflow: opening a catalog, spatial + row filtering, crossmatching, and writing results. References margin caches without explaining them (deferred to the margins tutorial).
- **Notable:** Several screenshots referenced but not visible in source (`ztf_catalog_lazy.png`, etc.); margin cache concept introduced without definition.

---

### `tutorials.rst`
**Title:** Tutorials  
**Role:** Navigation index; links to 8 topic-area toc files.  
Minimal prose; organizes tutorials into: Catalogs, Analyzing, Nested data, HATS creation, Performance, Rubin, Science examples, Debugging.

---

### `data-access.rst`
**Title:** Data Access  
**Role:** Overview and navigation for data access methods.  
High-level overview of how LSDB catalogs are hosted and accessed. Links to sub-pages for data.lsdb.io, external data centers, Hugging Face, HATS structure, remote access, TAP, and server-side filtering.

---

### `data-access/datalsdb.rst`
**Title:** Data Access via data.lsdb.io  
**Role:** User guide for the main data portal.  
Explains the catalog browser interface: hosting regions (US-WEST HTTP, US-EAST S3, US-WEST S3, Europe HTTP), access types, provider info (UW, STScI, IPAC/IRSA, CDS), and HATS versioning. Notes that catalog collections require HATS builder >= 0.6.0 (no explanation of what changed).

---

### `data-access/external.rst`  
**Title:** Data Access via External Data Centers  
**Role:** Pointer to external providers.  
Very brief. Lists LIneA (Brazil) and CDS (France). Notes that CDS hosts experimental HATS-on-the-fly infrastructure that may be slower and has limited feature support. Delegates detail to external sites.

---

### `data-access/hats.rst`
**Title:** HATS Catalog Structure and Performance  
**Role:** Technical reference for catalog layout and performance.  
Explains HEALPix spatial partitioning, Parquet storage, adaptive tiling, catalog collections, margin caches, index tables, association tables, skymaps, and metadata files. Covers performance implications: partition selection, random access, column pruning, metadata scans.
- **Notable:** Dense technical content; could benefit from a directory-structure diagram.

---

### `data-access/hats-huggingface.rst`
**Title:** HATS Catalogs on Hugging Face  
**Role:** Quick reference for Hugging Face-hosted catalogs.  
Explains the Multimodal Universe collection, `hf://` URI scheme, and huggingface_hub installation. Short; assumes users know Hugging Face basics.

---

### `data-access/server-lsdb.rst`
**Title:** server.lsdb.io (Under Construction)  
**Role:** Placeholder.  
Experimental server-side filtering service. No actual documentation content yet.

---

### `data-access/tap-lsdb.rst`
**Title:** tap.data.lsdb.io (Under Construction)  
**Role:** Placeholder.  
Experimental TAP endpoint for catalog discovery and querying. No actual documentation content yet.

---

### `data-access/remote_data.ipynb`
**Title:** Accessing Remote Data  
**Role:** How-to guide for remote catalog access (notebook).  
**Difficulty:** Intermediate  
Covers fsspec/universal_pathlib backends for HTTP/S, AWS S3, and other filesystems. Shows `storage_options` patterns and `UPath` usage. Has a substantial SSL troubleshooting section for specific platforms (Red Hat 8, uv package manager).
- **Notable:** Mostly configuration snippets rather than executable code; SSL section dominates.

---

### `citation.rst`
**Title:** About  
**Role:** License and citation.  
BSD 3-Clause license. Two citation papers: Caplar et al. 2025 (general LSDB) and Malanchev et al. 2025 (Rubin DP1). Funding: Schmidt Sciences, NSF, DIRAC Institute.

---

### `contact.rst`
**Title:** Contact Us / Getting Help  
**Role:** Community engagement hub.  
GitHub issues, two Slack channels (#lincc-frameworks-lsdb, #lincc-frameworks-qa), LSST Community Forum, drop-in office hours (Fridays for HATS/LSDB, Thursdays for LINCC Frameworks), working groups. Embedded Google Calendar.
- **Notable:** Slack requires invitation; calendar may not render in all doc formats.

---

### `developer/contributing.rst`
**Title:** Contributing to LSDB  
**Role:** Developer onboarding.  
Source installation, GitHub issue workflow, branch naming (`issue/##/description`), pytest, Sphinx docs building, pre-commit/black-jupyter hooks, PR process, release requests, tutorial contribution guidelines. Mentions pre-commit Python version mismatch as a known pain point.

---

### `tutorials/dask-cluster-tips.rst`
**Title:** Dask Cluster Configuration Tips  
**Role:** Practical deployment how-to.  
Covers LocalCluster setup, worker/thread tuning, memory management (managed vs. unmanaged allocations), multi-node clusters, SLURM via dask-jobqueue (Pittsburgh Supercomputing Center example), and Dask Dashboard monitoring. Includes PyArrow string conversion config.
- **Notable:** SLURM example is PSC-specific; GIL explanation is brief.

---

### `tutorials/dask-messages-guide.rst`
**Title:** Troubleshooting Frequent Dask Problems  
**Role:** Troubleshooting reference.  
Covers common Dask errors in order of severity: port-in-use warnings, dashboard access on clusters, worker pausing (80% memory), stream closed (harmless), unmanaged memory warnings, "poison pill" rescheduling (critical low-memory signal), worker death, task graph build delays.
- **Notable:** "Poison pill" is the most critical message but explained somewhat obliquely; action items could be clearer.

---

### `tutorials/kubernetes-deployment.rst`
**Title:** Kubernetes Deployment with the Dask Operator  
**Role:** Enterprise/large-scale deployment guide.  
Kubernetes prerequisites (v1.25+, kubectl, Helm), custom Docker image building, Dask Kubernetes Operator via Helm, DaskCluster YAML resources, PersistentVolume setup, connecting from inside/outside cluster, resource sizing by catalog scale, Dashboard monitoring, troubleshooting (OOM, idle workers, networking).
- **Notable:** Assumes Kubernetes/Helm familiarity; resource sizing recommendations are helpful.

---

### `tutorials/performance.rst`
**Title:** Performance  
**Role:** Benchmarking and competitive analysis.  
Crossmatching overhead measured at 5–15% above raw I/O. Benchmarks LSDB vs astropy and smatch on ZTF DR14 × Gaia DR3. Shows near-linear scaling with data size, out-of-memory support, and parallelization benefits. Hardware: Bridges2, 128 cores, 256 GB.
- **Notable:** Results from 2024–2025; hardware-specific; GitHub link for full analysis code included.

---

### `tutorial_toc/toc_*.rst` (8 files)
**Role:** Navigation-only index files. No content beyond toctree directives.  
- `toc_catalogs.rst` — Catalog object, lazy ops, column/row/region filtering, margins, custom search
- `toc_analyzing.rst` — Dask client, crossmatching, map_partitions, timeseries, plotting, exporting
- `toc_nested.rst` — NestedFrame, exploding lightcurves
- `toc_hats.rst` — Import catalogs, manual verification
- `toc_performance.rst` — Index tables, Dask cluster, Kubernetes, performance, joins, scaling workflows
- `toc_rubin.rst` — DP1 access, Rubin catalog ops, photo-z, VSX cross-match, periodic LC visualization
- `toc_science.rst` — ZTF×NGC, DES×Gaia, ZTF×PS1, DP1×Gaia epoch propagation, ZTF SN search
- `toc_debugging.rst` — Dask messages guide only

---

## Core Tutorial Notebooks

These live in `tutorials/` and are intended to be run by the user (no pre-executed outputs committed).

### `tutorials/catalog_object.ipynb`
**Title:** The Catalog Object  
**Difficulty:** Beginner  
Introduces the `Catalog` object: loading with `open_catalog()`, inspecting metadata (columns, schema, partitions, sky coverage, statistics), column pre-selection for efficiency, margin caches overview, `plot_pixels()` for partition visualization.
- **Notable:** Primarily expository; relies heavily on cross-references to other tutorials. Thin on guidance about which columns to keep.

---

### `tutorials/lazy_operations.ipynb`
**Title:** Lazy Operations in LSDB  
**Difficulty:** Intermediate  
Explains the lazy evaluation model: task graphs, metadata-only loading, Dask worker distribution, `compute()` for execution, `head()` for previewing. Emphasizes scalability from laptop to supercomputer.
- **Notable:** Uses embedded video/diagrams more than executable code. No explanation of what task-graph optimizations actually happen.

---

### `tutorials/column_filtering.ipynb`
**Title:** Column Filtering  
**Difficulty:** Beginner  
Three patterns for column selection on `open_catalog()`: default columns, `columns="all"`, and custom subset. Shows memory/performance benefit of pre-selecting columns.
- **Notable:** Very brief; doesn't discuss how to decide which columns are needed.

---

### `tutorials/row_filtering.ipynb`
**Title:** Row Filtering  
**Difficulty:** Beginner–Intermediate  
Two filtering approaches: `.query()` string expressions and boolean indexing with `[]`. Covers f-string variable injection, bitwise vs. logical operators, parentheses precedence, and preview methods (`head()`, `tail()`, `sample()`, `random_sample()`).
- **Notable:** Good progression simple→complex; covers common pitfalls.

---

### `tutorials/region_selection.ipynb`
**Title:** Region Selection  
**Difficulty:** Intermediate  
Four spatial filter types: cone search, box search, polygon search, MOC-based. Explains coarse (pixel) vs. fine (point) filtering stages, the `fine` parameter, and reusable Search objects. Includes `plot_pixels()` visualization.
- **Notable:** MOC construction deferred to external HATS docs; no discussion of when `fine=False` is appropriate.

---

### `tutorials/margins.ipynb`
**Title:** Margins  
**Difficulty:** Intermediate–Advanced  
Explains partition boundary incompleteness for spatial joins, margin caches as the solution, margin threshold parameter, automatic margin loading in Catalog Collections, explicit `margin_cache=` parameter, margin visualization, and duplicate avoidance.
- **Notable:** Good visualizations; `require_right_margin=False` workaround mentioned but not explained fully.

---

### `tutorials/dask_client.ipynb`
**Title:** Setting up a Dask Client  
**Difficulty:** Beginner–Intermediate  
Context-manager vs. persistent client patterns. Key arguments: `n_workers`, `threads_per_worker`, `memory_limit`, `local_directory`. Covers avoiding thread contention.
- **Notable:** No comparison of execution with vs. without a client; no cluster vs. local guidance.

---

### `tutorials/import_catalogs.ipynb`
**Title:** Importing Catalogs to HATS Format  
**Difficulty:** Intermediate  
Two ingestion paths: `lsdb.from_dataframe()` for small datasets (<1–2 GB) and `hats-import` map-reduce pipeline for large datasets (1 GB to 100+ TB). Parameters: `lowest_order`, `highest_order`, `partition_rows`. Uses test data with no scientific value.
- **Notable:** Guidance on choosing HealPix order parameters is minimal.

---

### `tutorials/small_scale.ipynb` *(incoming, not yet merged)*
**Title:** Small-Scale Analysis  
**Difficulty:** Intermediate  
Teaches the "start small, scale up" development workflow: use `cone_search` to restrict to a sky region, inspect individual partitions with `.partitions[i]`, apply row filters, use `map_partitions()` and `random_sample()` for cross-partition validation, and convert partition indices to sky coordinates via `get_healpix_pixels()`. Practical emphasis on avoiding expensive full-catalog runs during development.
- **Notable:** Fills a real gap — this methodology is crucial for large-catalog work but currently scattered across other notebooks. Strong candidate for early placement in a beginner path.

---

### `tutorials/exporting_results.ipynb`
**Title:** Exporting Results  
**Difficulty:** Beginner–Intermediate  
`write_catalog()` method, required parameters, HATS output directory structure (`Norder/Dir/Npix`), `catalog_name`, `default_columns`, resume/overwrite options, `storage_options` for cloud.
- **Notable:** No executable examples; all code is illustrative snippets only.

---

## Pre-executed: Technique Notebooks

Live in `tutorials/pre_executed/`. Outputs are committed; notebooks do not re-run in CI.

### `pre_executed/access.pyvo.ipynb`
**Title:** Finding HATS Catalogs via VO  
**Difficulty:** Beginner  
Uses pyVO to query the IVOA registry, filter results to HATS catalogs, and identify the fastest data mirror by testing access times.
- **Notable:** Very short (3 main cells); no explanation of reliability vs. access-time tradeoff.

---

### `pre_executed/nestedframe.ipynb`
**Title:** Understanding the NestedFrame  
**Difficulty:** Intermediate  
Comprehensive intro to nested-pandas `NestedFrame`: sub-column dot-notation access, filtering nested data, `map_rows()` for custom row analysis, the `.nest` accessor, `nest_lists()` for creating nested columns from lists, `join_nested()` from flat tables.
- **Notable:** Uses toy data; no real-astronomy example; limited discussion of memory/performance characteristics.

---

### `pre_executed/explode_lightcurves.ipynb`
**Title:** Convert Nested Lightcurves to Flat Source Tables with `explode`  
**Difficulty:** Beginner–Intermediate  
Shows `explode()` method (from nested-pandas) applied via `map_partitions` to flatten nested light-curve columns into a flat table.
- **Notable:** Very brief; no discussion of memory implications or when flattening is preferable to staying nested.

---

### `pre_executed/map_partitions.ipynb`
**Title:** map_partitions  
**Difficulty:** Intermediate–Advanced  
Comprehensive guide to user-defined functions across partitions: augmenting vs. reducing operations, `meta=` parameter for Dask metadata, `make_meta` helper, Dask client setup, accessing HEALPix pixel info. Includes memory/timing output from real infrastructure.
- **Notable:** No error-handling guidance for custom functions; memory profiling caveats flagged.

---

### `pre_executed/timeseries.ipynb`
**Title:** Working with Time Series Data  
**Difficulty:** Intermediate–Advanced  
Full time-series workflow: `nest_lists()` for ZTF-style list columns, nested filtering, `map_rows()` for per-object statistics, Lomb-Scargle periodograms (astropy), phase-folding, quality-flag filtering, handling ragged data.
- **Notable:** ZTF-specific structure assumed; cone search limits to one partition for demonstration.

---

### `pre_executed/visualize_periodic_lcs.ipynb`
**Title:** Visualizing Periodic Lightcurves  
**Difficulty:** Advanced  
Crossmatches known variable objects to Rubin DP1, extracts light curves from three data products (science imaging, difference imaging, forced photometry), converts flux to magnitudes with uncertainty propagation, phase-folds using known periods, visualizes with color-coded multi-band plots.
- **Notable:** Rubin DP1-specific; not easily adapted to other surveys; large number of custom plotting helpers.

---

### `pre_executed/plotting.ipynb`
**Title:** Plotting Results  
**Difficulty:** Intermediate  
LSDB visualization: `plot_pixels()` HEALPix maps, `plot_points()` scatter with WCS projections, color mapping, field-of-view control, multi-catalog overlays, custom aggregation functions for pixel-level plots, histograms.
- **Notable:** Some plots omitted from output; no interactive plotting; limited guidance on choosing appropriate visualization type.

---

### `pre_executed/custom_search.ipynb`
**Title:** Creating a Custom Search with Global Statistics  
**Difficulty:** Intermediate–Advanced  
Uses per-partition Parquet metadata statistics for efficient coarse filtering before touching actual data. Demonstrates implementing a custom `AbstractSearch` subclass for numeric extrema filtering.
- **Notable:** Notebook acknowledges the example code is "more finicky than typical use"; no guidance on performance tradeoffs.

---

### `pre_executed/index_table.ipynb`
**Title:** Using Index Tables  
**Difficulty:** Beginner–Intermediate  
Secondary index tables for ID-based lookup (vs. spatial lookup). Uses Gaia DR3 source IDs. Shows `id_search()`, catalog collections, and discovering available indexes.
- **Notable:** No guidance on creating index tables; no comparison of ID vs. spatial search performance.

---

### `pre_executed/manual_verification.ipynb`
**Title:** Manual Catalog Verification  
**Difficulty:** Beginner  
`is_valid_catalog()` with quick/strict modes and verbose/fail_fast flags. Schema inspection via PyArrow, column statistics aggregation, data type discovery.
- **Notable:** No guidance on what to do when validation fails; strict vs. loose tradeoffs not explained.

---

### `pre_executed/scaling_workflows.ipynb`
**Title:** Determining the Right Dask Cluster Parameters  
**Difficulty:** Advanced  
Profiles a real light-curve period analysis (Lomb-Scargle via `light_curve` package) to estimate cluster needs. Uses per-partition statistics to find largest partitions, memory-profiles the UDF, applies 1.5–2× overhead heuristic for cluster sizing. Shows single-threaded vs. multi-threaded worker tradeoffs.
- **Notable:** Requires `light-curve` and `memory-profiler`; Jupyter memory profiling caveats are substantial; numbers not guaranteed to generalize.

---

## Pre-executed: Science Example Notebooks

### `pre_executed/crossmatching.ipynb`
**Title:** Crossmatching Catalogs  
**Difficulty:** Intermediate  
Comprehensive crossmatch tutorial using ZTF DR22 × Gaia DR3. Covers: effect of catalog ordering on results (16 vs. 47 matches depending on left/right), KdTreeCrossmatch algorithm, margin cache importance, N-neighbors parameter, result verification via set operations, `plot_pixels()` and `plot_points()`.
- **Notable:** Demonstrates that ordering matters significantly; flags FutureWarning about suffix behavior.

---

### `pre_executed/join_catalogs.ipynb`
**Title:** Joining Catalogs  
**Difficulty:** Beginner–Intermediate  
Identifier-based join (vs. spatial crossmatch) using Gaia DR3 × Gaia EDR3 on `source_id`. Shows suffix handling for duplicate columns, computing derived quantities across catalog versions, and histogram comparison of parallax-based distances.
- **Notable:** Short and focused; histogram outlier (spike at 147) not investigated; FutureWarning about suffix behavior (same as crossmatching notebook).

---

### `pre_executed/des-gaia.ipynb`
**Title:** Cross-matching of Large Catalogs: DES to Gaia  
**Difficulty:** Advanced  
Full end-to-end pipeline: download DES DR2 (FITS) and Gaia DR3 (ECSV), convert both to HATS via `hats-import`, crossmatch at scale with LSDB, save results. Uses a data subset to keep runtime tractable; marked `execute: never` in notebook metadata.
- **Notable:** Requires `hats-import` beyond core LSDB; most realistic large-scale workflow example in the docs.

---

### `pre_executed/ztf_bts-ngc.ipynb`
**Title:** Cross-match ZTF BTS and NGC  
**Difficulty:** Intermediate  
Loads non-HATS ZTF Bright Transient Survey data via HTTP download, converts coordinates (hourangle → degrees), filters by redshift, crossmatches with NGC at 1200 arcsec, visualizes results on PanSTARRS images. Finds SN2022xxf in NGC 3705.
- **Notable:** Explicitly acknowledges NGC is "too shallow" for reliable matching; visualization portion (PanSTARRS) is peripheral to LSDB; requires internet access to multiple APIs.

---

### `pre_executed/ztf-alerts-sne.ipynb`
**Title:** Search for SN-like Light Curves in ZTF Alerts  
**Difficulty:** Intermediate–Advanced  
Downloads ~21 GB remote ZTF alert catalog with nested light curves. Applies Bazin parametric fitting and observation-count/chi-squared filters to identify 796 SN candidates.
- **Notable:** Very large download; requires `light-curve` package; filtering criteria may need adjustment for other science goals.

---

### `pre_executed/zubercal-ps1-snad.ipynb`
**Title:** Get Light Curves from ZTF and PS1 for SNAD Catalog  
**Difficulty:** Intermediate  
Loads a custom SNAD pointing catalog as a DataFrame, crossmatches against PS1 DR2 (S3) and ZTF DR16 Zubercal (HTTPS) without pre-download, joins nested catalogs, visualizes multi-band light curves (ZTF g/r/i and PS1 g/r/i/z/y).
- **Notable:** Shows remote access without downloading; acknowledges some SNAD objects have no PS1 data; takes 5+ minutes for joins across remote datasets.

---

## Pre-executed: Rubin-specific Notebooks

### `pre_executed/rubin_dp1.ipynb`
**Title:** Accessing Rubin Data Preview 1 (DP1)  
**Difficulty:** Intermediate  
Step-by-step access guide across four compute platforms: RSP (Rubin Science Platform), NERSC (Perlmutter), CANFAR, and LIneA. Shows how to open `object_collection` and `dia_object_collection` with LSDB on each.
- **Notable:** NERSC section shows LSDB 0.6.7 while RSP section shows 0.9.0 — platforms are not at the same version. No troubleshooting for common access failures. No guidance on choosing between platforms.

---

### `pre_executed/using_rubin_data.ipynb`
**Title:** Intro to Rubin Catalog Operations  
**Difficulty:** Beginner–Intermediate  
Most accessible Rubin notebook. Covers column discovery (`all_columns`, `nested_columns`), light-curve visualization with `lsdb-rubin` plotting utilities, filtering by nDiaSources and band, handling empty light curves after filtering, computing custom statistics with `map_rows()`.
- **Notable:** Good entry point for Rubin users; warns that `nDiaSources` becomes stale after filtering (but this warning is brief and easy to miss).

---

### `pre_executed/rubin_dp1_photoz.ipynb`
**Title:** RAIL Photo-z Estimates for Rubin DP1  
**Difficulty:** Advanced  
Works with photo-z PDFs from 8 algorithms (Lephare, kNN, TPZ, BPZ, CMNN, GPZ, DNF, FZBoost). Loads catalog for data-rights holders via LSDB; provides a public Parquet fallback. Reconstructs `qp.Ensemble` objects from nested PDF data in different representations (interp, mixmod, norm, hist, quantile).
- **Notable:** qp ensemble reconstruction is complex and "very computationally expensive" on full dataset but demonstrated on only 5 objects. No guidance on choosing among 8 algorithms. Plotting only shows two of the eight.

---

### `pre_executed/rubin_dp1_vsx.ipynb`
**Title:** Cross-matching Rubin DP1 and Variable Star Index (VSX)  
**Difficulty:** Intermediate  
Loads DP1 object catalog and VSX from AAVSO, filters VSX to eclipsing binary types (EA, EB, EW), performs 0.1 arcsec spatial crossmatch, finds 9 matches.
- **Notable:** Only 9 matches found — no discussion of whether this is expected or a completeness issue. No visualization of results. Cross-match radius choice unexplained.

---

### `pre_executed/dp1-gaia-epoch-prop.ipynb`
**Title:** Astrometric Epoch Propagation  
**Difficulty:** Advanced  
Propagates Gaia DR3 positions from epoch 2016.0 to DP1 epoch (~2024.9) using proper motion, parallax, and RV with astropy. Compares propagated (1492 matches) vs. naive (957 matches) crossmatches on DP1, showing ~36% improvement. Uses safe search radius derived from DP1 margins.
- **Notable:** Why RV-only sources are selected is not explained. Improvement is significant but discussion is brief. Complex custom matching logic with minimal comments.

---

## Cross-cutting Observations

### What's working well
- Comprehensive coverage of deployment scenarios (local, HPC SLURM, Kubernetes).
- Strong troubleshooting content (Dask messages guide, dask-cluster-tips).
- Science examples span multiple surveys and science cases (transients, variable stars, photo-z, astrometry).
- Clear contribution guidelines and citation information.
- Performance benchmarks with methodology and code links.

### Structural gaps / issues
- **Navigation fragmentation:** Eight `toc_*.rst` files add an indirection layer without adding content. The tutorial structure is hard to follow from the top.
- **Two "Under Construction" stubs** (`server-lsdb.rst`, `tap-lsdb.rst`) visible to users with no content.
- **Beginner entry point is thin:** `getting-started.rst` is short and references concepts (margin caches) without defining them. There is no single notebook a brand-new user should run first.
- **`catalog_object.ipynb` and `lazy_operations.ipynb`** are mostly expository with few runnable cells, making them weak as "first notebooks."
- **Rubin notebooks are split across two sections** (toc_rubin + individual entries in toc_science/toc_analyzing) with no clear reading order for a Rubin user.
- **`exporting_results.ipynb`** has no executable cells — effectively a reference doc masquerading as a notebook.
- **Dask client setup** is repeated in many notebooks without pointing to the canonical `dask_client.ipynb`.
- **`hats.rst`** (HATS structure) is under data-access but is really conceptual background that many tutorials implicitly require.

### Notebook difficulty spread
| Difficulty | Count | Key examples |
|---|---|---|
| Beginner | ~5 | catalog_object, column_filtering, row_filtering, access.pyvo, manual_verification |
| Beginner–Intermediate | ~6 | dask_client, exporting_results, join_catalogs, using_rubin_data, explode_lightcurves, index_table |
| Intermediate | ~8 | lazy_operations, region_selection, crossmatching, ztf_bts-ngc, zubercal-ps1-snad, plotting, rubin_dp1, rubin_dp1_vsx |
| Intermediate–Advanced | ~5 | map_partitions, timeseries, ztf-alerts-sne, custom_search, margins |
| Advanced | ~6 | des-gaia, scaling_workflows, dp1-gaia-epoch-prop, rubin_dp1_photoz, visualize_periodic_lcs, import_catalogs |

### Audience coverage
- **Brand-new LSDB user path** exists but is scattered and requires knowing where to look.
- **Rubin-specific users** have their own section but no suggested reading order and some content is split into other sections.
- **Power users / developers** are well-served by performance, scaling, Kubernetes, and contributing docs.
- **Reference lookup** (e.g., "how do I do a polygon search?") is reasonably discoverable but the API reference (autoapi) is separate from the prose tutorials.
