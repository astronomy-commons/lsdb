# LSDB Documentation Restructuring Proposal

> Working document for team discussion. Companion to `_doc_inventory.md`.

---

## Background

This proposal is the result of a full read-through of all 33 notebooks and 25 prose docs that render on docs.lsdb.io. The goal is to serve two audiences better:

1. **New users** — experienced astronomers (PhD-level assumed) who have never used LSDB and need to get productive quickly without being talked down to.
2. **Returning users** — everyday users who need to find something specific: a code pattern, a method they haven't used in a while, an explanation of a feature they're not sure about.

These two audiences have different needs and currently the docs serve neither particularly well, despite the content itself being solid.

---

## What We Found

### Strengths worth preserving
- The notebook content is generally high quality and well-calibrated for the target audience.
- Science example notebooks (DES×Gaia, ZTF×NGC, ZTF SN search, etc.) are strong and practically useful.
- Troubleshooting content (Dask messages guide, cluster tips) is thorough.
- Performance benchmarks include methodology and code links.
- The Rubin section is timely and well-populated.

### Structural problems

**1. No guided entry point.**
`getting-started.rst` covers installation and a code sketch, but references "margin caches" without defining them and relies on screenshots that may not render. There is no single notebook a new user should run first. `catalog_object.ipynb` and `lazy_operations.ipynb` are the most beginner-facing notebooks but are mostly expository prose with few runnable cells — a weak landing.

**2. The 8-file TOC is fragmented without adding clarity.**
`toc_catalogs`, `toc_analyzing`, `toc_nested`, `toc_hats`, `toc_performance`, `toc_rubin`, `toc_science`, `toc_debugging` — each is a nearly-empty file whose only content is a toctree directive. This adds a navigation layer without adding orientation. Users browse a list of category names, then another list of notebook titles, with no summary of what each contains or why they'd want it.

**3. No recommended reading order.**
The docs implicitly expect users to browse and self-select. That works for returning users with a specific goal, but not for someone who doesn't yet know what they need. Nothing says "start here, then go here."

**4. `small_scale.ipynb` has no home.**
This incoming notebook (teaching the "develop small, scale up" workflow) addresses one of the most common stumbling blocks for new large-catalog users. It doesn't appear in any existing TOC category, and it belongs very early in a beginner path.

**5. Rubin content is split across two sections.**
The Rubin section (`toc_rubin`) holds access and introductory notebooks, but some Rubin science notebooks live in `toc_science` and `toc_analyzing`. A Rubin user has to know to look in multiple places.

---

## Proposed Changes

### 1. Add a genuine "Getting Started" section

Replace the thin `getting-started.rst` with a short, explicitly sequential path of 4–5 pages/notebooks. A new user follows this top-to-bottom before going anywhere else. Suggested sequence:

| Step | Content | Source |
|------|---------|--------|
| 1 | **What is LSDB?** — what problem it solves, where HATS fits in, when to use it | Existing `index.rst` / `getting-started.rst` content (fine as-is) |
| 2 | **Installation** — environment setup, pip/conda install | `getting-started.rst` (trim) |
| 3 | **Open and explore a catalog** — `open_catalog()`, metadata inspection, lazy model | `catalog_object.ipynb` |
| 4 | **Develop at small scale** — cone search, single-partition inspection, sampling; the "work small, scale up" habit | `small_scale.ipynb` (incoming) |
| 5 | **Where to go next** — brief "if you're doing X, see Y" guide to the rest of the docs | New short prose page |

This section is **sequential and concept-building**: each step assumes the previous one. It is distinct from the main Tutorials section, which is organized by operation and can be entered at any point.

---

### 2. Consolidate the 8 TOC files into 4 sections

The main tutorial area stays **organized by operation** (what you're doing with catalogs). Proposed 4 sections replace the current 8:

#### Section A: Filtering and Selecting Data
Covers all the ways to narrow down what you're working with.

- Column filtering (`column_filtering.ipynb`)
- Row filtering (`row_filtering.ipynb`)
- Region selection (`region_selection.ipynb`)
- Margins (`margins.ipynb`)
- Lazy operations (`lazy_operations.ipynb`)
- Custom search (`custom_search.ipynb`)

#### Section B: Combining Catalogs
Covers all the ways to relate two or more catalogs.

- Crossmatching catalogs (`crossmatching.ipynb`)
- Joining catalogs (`join_catalogs.ipynb`)
- Index tables (`index_table.ipynb`)

#### Section C: Nested and Time-Series Data
Covers the nested-pandas layer and time-series workflows.

- Understanding NestedFrame (`nestedframe.ipynb`)
- Working with time series (`timeseries.ipynb`)
- Exploding lightcurves (`explode_lightcurves.ipynb`)

#### Section D: Infrastructure and Performance
Covers compute setup, scaling, and profiling.

- Setting up a Dask client (`dask_client.ipynb`)
- Dask cluster configuration tips (`dask-cluster-tips.rst`)
- Scaling workflows (`scaling_workflows.ipynb`)
- Kubernetes deployment (`kubernetes-deployment.rst`)
- Troubleshooting Dask messages (`dask-messages-guide.rst`)
- Performance benchmarks (`performance.rst`)
- Plotting results (`plotting.ipynb`)

---

### 3. Add a "Getting Data In and Out" section

Currently these topics are split between `toc_hats`, `data-access/`, and scattered `toc_performance` entries. Consolidate:

- Importing catalogs to HATS (`import_catalogs.ipynb`)
- Exporting results (`exporting_results.ipynb`)
- Finding catalogs via VO (`access.pyvo.ipynb`)
- Accessing remote data (`remote_data.ipynb`)
- Manual catalog verification (`manual_verification.ipynb`)
- Data access overview (`data-access/` pages: data.lsdb.io, Hugging Face, external data centers, server.lsdb.io, tap.data.lsdb.io)

---

### 4. Consolidate the Rubin section

All Rubin content in one place, in a natural reading order. Any notebook that primarily uses Rubin data lives here rather than in general tutorials or science examples.

1. Accessing Rubin DP1 (`rubin_dp1.ipynb`)
2. Intro to Rubin catalog operations (`using_rubin_data.ipynb`)
3. Photo-z estimates (`rubin_dp1_photoz.ipynb`)
4. Cross-matching DP1 with VSX (`rubin_dp1_vsx.ipynb`)
5. Astrometric epoch propagation with Gaia (`dp1-gaia-epoch-prop.ipynb`) *(moved from science examples)*
6. Visualizing periodic lightcurves (`visualize_periodic_lcs.ipynb`) *(moved from science examples)*

---

### 5. Keep Science Examples as a standalone section

These are end-to-end demonstrations, neither tutorials nor how-tos. They should be clearly labeled as "see how LSDB is used for real research" and not mixed into the tutorial flow.

- Cross-match ZTF BTS and NGC (`ztf_bts-ngc.ipynb`)
- Import and cross-match DES and Gaia (`des-gaia.ipynb`)
- Get light curves from ZTF and PS1 (`zubercal-ps1-snad.ipynb`)
- Search for SN-like light curves in ZTF alerts (`ztf-alerts-sne.ipynb`)

*(dp1-gaia-epoch-prop and visualize_periodic_lcs moved to Rubin section)*

---

### 6. Document the pre-executed notebook convention in the contributor guide

The `pre_executed/` folder is a build constraint (notebooks that take too long or need unavailable environments for RTD to run), not a conceptual distinction — readers on RTD see all notebooks the same way. `contributing.rst` should document this clearly so future contributors know when to place a notebook in `pre_executed/` vs. `tutorials/`.

Additionally, `contributing.rst` should link to `developer/tutorial_template.ipynb`, which currently has no user-facing entry point.

---

## Before / After: Top-Level TOC

### Current structure
```
LSDB
├── Getting Started
├── Tutorials
│   ├── Catalogs (→ toc_catalogs.rst)
│   ├── Analyzing Catalogs (→ toc_analyzing.rst)
│   ├── Nested Data Manipulation (→ toc_nested.rst)
│   ├── HATS Creation and Reading (→ toc_hats.rst)
│   ├── Performance Tips (→ toc_performance.rst)
│   ├── Working with Rubin Data (→ toc_rubin.rst)
│   ├── Science Examples (→ toc_science.rst)
│   └── Debugging (→ toc_debugging.rst)
├── Data Access
├── Reference (API)
├── About / Cite
├── Contact
└── Contributing
```

### Proposed structure
```
LSDB
├── Getting Started                          ← new, sequential
│   ├── What is LSDB?
│   ├── Installation
│   ├── Open and Explore a Catalog
│   ├── Developing at Small Scale            ← small_scale.ipynb (incoming)
│   └── Where to Go Next
├── Tutorials
│   ├── Filtering and Selecting Data         ← replaces toc_catalogs + parts of toc_analyzing
│   ├── Combining Catalogs                   ← replaces crossmatch/join in toc_analyzing + index in toc_performance
│   ├── Nested and Time-Series Data          ← replaces toc_nested
│   └── Infrastructure and Performance      ← replaces toc_performance + toc_debugging
├── Getting Data In and Out                  ← consolidates toc_hats + data-access
├── Working with Rubin Data                  ← consolidates toc_rubin + Rubin entries from toc_science/toc_analyzing
├── Science Examples                         ← toc_science (non-Rubin entries only)
├── Reference (API)
├── About / Cite
├── Contact
└── Contributing
```

---

## What Stays the Same

- All notebook content (no rewrites proposed at this stage)
- The Data Access sub-pages and stubs
- The API reference section
- The Science Examples notebooks (content unchanged; two moved to Rubin section)

---

## Low Priority / Future Attention

The following notebooks have little or no runnable code and function more as reference pages than interactive tutorials. They are not a priority to change, but worth revisiting when capacity allows:

- `catalog_object.ipynb` — mostly expository; few runnable cells
- `lazy_operations.ipynb` — primarily prose and embedded media
- `exporting_results.ipynb` — no runnable cells; reads as a reference page

---

## Open Questions for Team Discussion

1. **Dask client setup boilerplate** — Many notebooks independently set up a Dask client. Should a cross-reference to `dask_client.ipynb` be added to notebooks that set up a client, or is the repetition acceptable for standalone usability?
