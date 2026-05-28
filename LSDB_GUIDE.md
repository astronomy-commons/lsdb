# LSDB Guide

**Last updated:** 2026-05-28 | **LSDB version:** v0.9.0

Canonical reference for AI coding assistants working on LSDB. Tool-specific files
(`CLAUDE.md`, `.github/copilot-instructions.md`) contain only tool-specific overrides
and reference this file for shared guidance. **Edit this file** for changes that should
apply to all AI assistants; edit tool-specific files only for tool-specific behavior.

> **Version notice for AI assistants:** If anything in this guide conflicts with what
> you observe in the actual code (missing methods, different signatures, changed
> behaviour), the guide may be outdated. Check the version above against the installed
> package version (`python -c "import lsdb; print(lsdb.__version__)"`) and **alert the
> user** about the changes that `LSDB_GUIDE.md` might need. Do not silently assume the
> guide is correct.

## What Is LSDB

LSDB is a Python library for scalable spatial analysis of large
astronomical catalogs, designed to handle order of billion sources from upcoming surveys like LSST,
Euclid, and Roman. It is built on Dask for distributed parallelization and uses the HATS
(Hierarchical Adaptive Tiling Scheme) partitioning format for efficient spatial operations.
LSDB handles the boilerplate around loading, querying, cross-matching, and transforming
sky catalogs so astronomers can focus on science.

## Design Goals and North Stars

**CRITICAL: Always keep these design principles in mind when making changes to LSDB.**

**Lazy evaluation by default.** All operations on a `Catalog` build a Dask task
graph and return immediately - nothing is computed until `.compute()` is called or a
result is written to disk. Never eagerly compute results inside library methods; preserve
laziness so users can chain operations before triggering execution.

**HATS partitioning is the backbone.** Every catalog is partitioned by HEALPix pixel and the
underlying pixel data is stored on disk in parquet. Cross-partition operations (crossmatching,
joins) require margin catalogs to avoid missing objects near partition edges.

**Catalog collections**: Catalogs can be stand-alone or part of a collection. A collection
groups related catalogs together (main, margin, and index catalogs). When loading a collection, the
default margin is loaded automatically and attached to the main catalog (`catalog.margin`).

**Margins matter for correctness.** Any spatial operation that can involve objects within
some radius of a partition boundary (crossmatch, cone search near edges) must use a
`MarginCatalog`. Do not silently drop margin handling - incorrect results at partition
edges are worse than a clear error.

**Slim API surface.** Do not add new public API methods unless asked. Prefer composing existing
primitives. If you think a new method is needed, propose it first and get an agreement on the design
before implementing. When adding a new method, ensure it is added to the docs API reference and has
a complete docstring with examples.

**Scale from laptop to cluster.** LSDB code must work on a single-machine local Dask scheduler
and on a distributed `dask.distributed` cluster without code changes. Scalability is critical. Avoid
patterns that only work on small datasets or require loading everything into memory. Use Dask best practices
to ensure good performance at scale.

**Backwards compatibility.** Maintain backward compatibility where possible! If breaking changes are
necessary, be loud about it.

**Document current behavior.** When migrating away from old patterns, use `@deprecated` with a helpful
message rather than silently removing behavior. Error messages should point users to the documentation
or the correct alternative. 

**Docstrings and type safety.** All public methods must have complete NumPy-style docstrings and 
accurate type annotations.

## Coding advice

- **Do not push or open PRs** unless explicitly asked.
- When changing code, ensure that the current assumptions of the change appear to have always 
been true. 
- Leave code better than you find it over keeping old assumptions around.

## Development Setup

- **Python ≥ 3.11** (see `pyproject.toml` `requires-python`)
- Create a conda env: `conda create -n lsdb python=3.11 && conda activate lsdb`
- Clone and install: `git clone https://github.com/astronomy-commons/lsdb.git && cd lsdb`
- Run the setup script: `echo 'y' | bash .setup_dev.sh`
  - Installs the package in editable mode with dev extras
  - Installs pre-commit hooks
- Alternative manual install: `pip install -e .'[dev]' && pre-commit install`
- For full optional features (e.g. plotting, polygon search, Lance): `pip install -e '.[full]'`
- For S3 / object-store support: `pip install -e '.[s3fs]'`
- For bleeding-edge dependency versions (hats + nested-pandas from `main`): `pip install -r requirements.txt`
- For documentation dependencies: `pip install -r docs/requirements.txt`

## Common Commands

```bash
# Run the full test suite (includes doctests in src/ and docs/)
python -m pytest

# Run only unit tests (skip doctest collection from docs/)
python -m pytest tests/

# Run with coverage reporting
python -m pytest --cov=lsdb --cov-report=xml

# Lint
pylint src/ --rcfile=./src/.pylintrc
pylint tests/ --rcfile=./tests/.pylintrc

# Format
black src/ tests/ && isort src/ tests/

# Type check
mypy src/ tests/ --ignore-missing-imports

# Pre-commit (runs black, isort, pylint, mypy ...)
pre-commit run --all-files

# Build docs. Requires `docs/requirements.txt` dependencies installed.
cd docs && make html

# Run ASV benchmarks
cd benchmarks && asv run --quick
```

## Repository Structure

```
src/lsdb/               Main package
src/lsdb/catalog/       Catalog class hierarchy (Catalog, MarginCatalog, MapCatalog, AssociationCatalog)
src/lsdb/core/          Core algorithms (crossmatch, spatial search, plotting)
src/lsdb/dask/          Dask-level operations and task graph construction
src/lsdb/loaders/       Catalog loading (from HATS, pandas DataFrame, astropy Table)
src/lsdb/io/            Write paths (to_hats, to_lance, to_association)
src/lsdb/nested/        NestedFrame: Dask DataFrame extension for nested data
src/lsdb/streams/       CatalogStream for iterating over partitions in chunks
tests/lsdb/             Test suite (mirrors src/ layout)
tests/data/             Small HATS-formatted test catalogs and other validation files
benchmarks/             ASV performance benchmarks
docs/                   Sphinx documentation sources
docs/reference/         API reference (auto-generated from docstrings)
docs/tutorials/         Jupyter notebook tutorials
```

Key files:

| File                                                        | Purpose                                                        |
|-------------------------------------------------------------|----------------------------------------------------------------|
| `pyproject.toml`                                            | Project metadata, dependencies, pytest/black/mypy config       |
| `src/lsdb/__init__.py`                                      | Public API - everything exported here is stable public surface |
| `src/lsdb/catalog/catalog.py`                               | Main `Catalog` class                                           |
| `src/lsdb/catalog/dataset/healpix_dataset.py`               | Base `HealpixDataset` class, shared by all catalog types       |
| `src/lsdb/loaders/hats/read_hats.py`                        | `open_catalog` and `read_hats` entry points                    |
| `src/lsdb/loaders/dataframe/from_dataframe.py`              | `from_dataframe` and `from_astropy` entry points               |
| `src/lsdb/core/crossmatch/crossmatch.py`                    | Top-level `crossmatch()` function                              |
| `src/lsdb/core/crossmatch/abstract_crossmatch_algorithm.py` | Abstract class to implement custom crossmatch algorithms       |
| `src/lsdb/core/search/abstract_search.py`                   | Abstract class to implement custom spatial searches            |
| `src/lsdb/dask/`                                            | Internal Dask graph construction - not public API              |

## Architecture: Catalog Class Hierarchy

All catalog types inherit from `HealpixDataset` and carry three core attributes:

- `ddf` - a `NestedFrame` (Dask DataFrame with nested-pandas extension)
- `ddf_pixel_map` - `Dict[HealpixPixel, int]` mapping HEALPix pixels to Dask partition indices
- `hc_structure` - the corresponding `hats` metadata object (no actual data)

### Catalog (`src/lsdb/catalog/catalog.py`)
The primary user-facing class. Wraps a HATS-partitioned sky catalog.

**Key method groups:**
- **Introspection:** `est_size`, `get_healpix_pixels`, `plot_coverage`, `plot_pixels`, `plot_points` 
- **Query and spatial search:** `cone_search`, `box_search`, `id_search`, `moc_search`, `pixel_search`, `polygon_search`, `query`, `search` (generic)
- **Cross-catalog operations:** `concat`, `crossmatch`, `crossmatch_nested`, `join`, `join_nested`, `merge`, `merge_asof`, `merge_map`
- **Partition and row level transforms:** `map_partitions`, `map_rows`
- **Column and pixel statistics:** `aggregate_column_statistics`, `per_partition_statistics`
- **Sampling (eager execution):** `head`, `tail`, `sample`, `random_sample`
- **Computing (eager execution):** `compute`, `write_catalog`

### MarginCatalog (`src/lsdb/catalog/margin_catalog.py`)
Holds objects from neighboring partitions within a specified angular radius of
each partition boundary. A `Catalog` has often a corresponding margin which can
be access via `catalog.margin`.

### MapCatalog (`src/lsdb/catalog/map_catalog.py`)
Represent non-point-source data (e.g. dust extinction, survey depth values) in a continuous map.

### AssociationCatalog (`src/lsdb/catalog/association_catalog.py`)
Represents the pre-computed result of a crossmatch or join operation. Carries the pixels to join to recreate the full
crossmatch/join result and any extra columns (e.g. object distance in arcseconds). Each association catalog is defined
by a `max_separation` (arcseconds) for which the association catalog is valid. When joining catalogs through an
association, the right catalog must have a margin threshold at least as high as the `max_separation` to ensure all
matches are captured.

## Architecture: Search and Crossmatch Extension Points

### AbstractSearch (`src/lsdb/core/search/abstract_search.py`)
Subclass to implement a custom spatial filter. Must implement:
- `filter_hc_catalog(hc_structure)` - return the subset of HEALPix pixels that
  could contain matching objects.
- `search_dataframe(df, catalog_info)` - filter a single-partition NestedFrame.

Pass an instance to `catalog.search(my_search)`.

### AbstractCrossmatchAlgorithm (`src/lsdb/core/crossmatch/abstract_crossmatch_algorithm.py`)
Subclass to implement a custom crossmatch algorithm and pass an instance to the crossmatch:
`catalog.crossmatch(other, algorithm=MyCustomAlgorithm())`. The default algorithm
is `KdTreeCrossmatch`, an efficient implementation using `scipy.spatial.cKDTree` for fast nearest neighbor search.

## Typical LSDB Workflow

A typical LSDB workflow involves:

1. **Opening catalogs** from an existing HATS catalog, `pd.DataFrame` or `astropy.Table`.
2. **Exploring metadata** to understand the data:
   - Size (length) of the catalog
   - Schema (columns available and their types)
   - Estimated size of the catalog
   - Sky coverage and partitioning structure
   - The columns/pixel statistics (e.g. mean, min, max) to understand the data distribution
3. **Performing spatial queries** (cone search, box search, polygon search) to subset the catalog to a region of interest.
4. **Crossmatching** with another catalog, ensuring margin handling for edge correctness.
5. **Transforming results** with `map_partitions` or `map_rows` for custom computations.
6. **Writing results** to HATS format for downstream use or sharing.

### Load catalogs

LSDB can handle local or remote catalogs with `fsspec`. All paths are transformed into `UPath` objects internally.

```python
import lsdb

# Open a HATS catalog / collection from disk or object store.
# When opening a collection, the default margin is loaded and attached to the main catalog (`cat.margin`).
cat = lsdb.open_catalog("/path/to/hats/catalog", columns=["ra", "dec", "flux_g", "flux_r"])

# Open an auxiliary catalog directly from disk or object store.
margin = lsdb.read_hats("/path/to/hats/margin")

# Load a small pandas DataFrame as a catalog (< 1M rows)
cat = lsdb.from_dataframe(df, ra_column="ra", dec_column="dec")

# Load from an astropy Table
cat = lsdb.from_astropy(table, ra_column="ra", dec_column="dec")

# Generate a synthetic test catalog
# 500 rows in the base layer, 10 rows in the nested layer
cat = lsdb.generate_catalog(500, 10, seed=1)
```

#### CRITICAL
- Only load the columns you need when running your analysis to minimize memory usage and speed up operations. 
- Use the `columns` argument in `open_catalog` to specify the desired subset of columns.

### Spatial queries
```python
# Cone search (returns a new Catalog, lazy)
result = cat.cone_search(ra=180.0, dec=-30.0, radius_arcsec=3600)

# Box search
result = cat.box_search(ra=(170, 190), dec=(-40, -20))

# Polygon search (requires lsst-sphgeom)
result = cat.polygon_search(vertices=[(300, -50), (300, -55), (272, -55), (272, -50)])
```

### Crossmatch
```python
# Basic crossmatch (inner join by default)
xmatch = cat_a.crossmatch(cat_b, radius_arcsec=1.0)

# Basic crossmatch (left join)
xmatch = cat_a.crossmatch(cat_b, radius_arcsec=1.0, how="left")

# Crossmatch with a custom algorithm
xmatch = cat_a.crossmatch(cat_b, radius_arcsec=1.0, algorithm=MyCustomAlgorithm())
```

`cat_b`'s `margin_threshold` (arcsec) must be greater or equal to `radius_arcsec`.

### Custom partition operations
```python
def partition_transform(df, *, inc):
    df["new_ra"] = df["ra"] + inc
    return df

result = cat.map_partitions(partition_transform, inc=1)
```

If `partition_transform` function does not handle the empty partition case, LSDB might fail to infer the result Dask meta. 
In that case, manually specify `meta`: `cat.map_partitions(..., meta=)`.

### Custom row operations
```python
def row_transform(ra, dec, mag):
    return {"ra": ra, "dec": dec, "num_src": len(mag)}

result = cat.map_rows(
    row_transform,
    columns=["ra", "dec", "sources.mag"],
    row_container="args",
    meta={"ra": float, "dec": float, "num_src": int},
    append_columns=False # default
)
```

Always specify `meta`. If `append_columns=True`, `meta` should only include the new columns returned by `row_transform`.

### Stream large catalogs
```python
from lsdb.streams import CatalogStream

stream = CatalogStream(cat, partitions_per_chunk=10)
for chunk in stream:
    process(chunk)
```

### Compute and write results

```python
from dask.distributed import Client

with Client(n_workers=4, threads_per_worker=1, memory_limit="4GB") as client:
  # Write to HATS (Parquet-based, re-readable by lsdb.open_catalog)
  cat.write_catalog("/output/path")

  # Write to Lance format
  cat.to_lance("/output/path")
  
  # Compute to memory (use only for small results)
  df = cat.compute()
```

#### CRITICAL
- Do not load a full catalog into memory with `.compute()` unless you are sure it fits in memory!
- Plan the computation by chaining lazy operations and only compute/write the results at the very end!
- Suggest the creation of a `dask.distributed.Client` to take advantage of distributed computation.
- Make sure the `Client` specifies appropriate defaults for the `n_workers` and `memory_limit` (per worker) 
according to the available resources.

## Troubleshooting

The Dask dashboard is the primary tool for diagnosing performance problems. It shows task
progress, worker memory, and the task stream in real time.

```python
from dask.distributed import Client
client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")
print(client.dashboard_link)  # e.g. http://127.0.0.1:8787/status
```

- The dashboard is available as soon as the `Client` is created. Open the URL in a browser.
- Use a custom port if 8787 is taken: `Client(..., dashboard_address=":8790")`.
- On a remote machine or JupyterHub, set up an SSH tunnel:
  `ssh -L 8787:localhost:8787 user@remote-host`, then open `http://localhost:8787/status`.

**Key dashboard panels and what to look for:**

- **Progress** - shows how many tasks are pending, running, and done per computation.
  A bar that barely moves means workers are idle or a single task is a bottleneck.
- **Task Stream** - each horizontal bar is a worker; colored blocks are tasks. Gaps
  between blocks indicate workers waiting on each other or on the scheduler. Aim for
  dense, uniform coverage across all workers.
- **Workers** - shows CPU utilization and memory per worker. Workers near their
  `memory_limit` will start spilling to disk (orange) or get restarted (red). If this
  happens, try to reduce `n_workers` and increase `memory_limit`.
- **Graph** - visualizes the full Dask task graph. The number of tasks tends to scale
  with the number of partitions.

### Error messages and failures

For a detailed guide to common Dask log messages, worker OOM scenarios, and memory
tuning strategies, see `docs/tutorials/dask-messages-guide.rst`. It covers:

- Worker paused at 80% memory usage
- `Couldn't gather N keys, rescheduling` (fatal silent OOM - restart the kernel)
- `Unmanaged memory use is high` warning
- `CommClosedError: Stream is closed` after a `with Client(...)` block (cosmetic)
- All workers killed at the start of a job (how to get a real Python traceback)
- All workers killed in the middle or end of a job (unbalanced task memory)
- Dashboard empty for a long time after `.compute()` (graph construction phase)

## Testing Conventions

- **File naming:** `tests/lsdb/test_<name>.py`, mirroring the `src/lsdb/` layout.
- **Fixtures:** defined in `tests/conftest.py`. Use existing fixtures, do not
  duplicate test data. All fixtures are backed by tiny HATS catalogs in `tests/data/`.
- Default test run: `python -m pytest`
- **Doctest enforcement:** `pytest` is configured with `--doctest-modules` and
  `--doctest-glob=*.rst`. All public docstring examples must be runnable and correct.
- **No network in unit tests.** Test data lives in `tests/data/`; do not fetch from
  the internet in unit tests.

## Key Conventions

- **Line length:** 110 characters (`black` and `isort` both enforce this).
- **Import style:** `isort` with `profile = "black"`. Do not hand-tune import order.
- **Docstrings:** NumPy style. All public functions and methods require a complete
  docstring including `Parameters` and `Returns`. Try to also include an `Examples` block.
- **Deprecation:** use `@deprecated(version="X.Y", reason="...")` from the
  `deprecated` package. Never silently remove behavior.
- **`_version.py` is auto-generated** by `setuptools_scm` from git tags. Never
  edit it by hand; it is excluded from coverage and linting.
- **`lsst-sphgeom` is platform-gated:** polygon search requires this package, which
  is only available on macOS and Linux (`sys_platform` guard in `pyproject.toml`).
  Code using `lsst-sphgeom` must guard the import and provide a clear error on Windows.
- **`dask/` is internal:** functions in `src/lsdb/dask/` are implementation details.
  Do not expose them in `__init__.py` or reference them in documentation.
- **Partition index vs. pixel:** `ddf_pixel_map` maps `HealpixPixel → int` (partition
  index). Never assume partition indices are contiguous or match pixel order.

## CI/CD and GitHub Workflows

- **`testing-and-coverage.yml`** - runs on every PR and push to `main`; matrix over
  Python 3.11–3.14; uploads coverage to Codecov.
- **`smoke-test.yml`** - daily at 06:45 UTC; tests both `[dev]` and `[full]` extras.
- **`testing-windows.yml`** - Windows-specific test matrix.
- **`asv-main.yml` / `asv-pr.yml` / `asv-nightly.yml`** - ASV performance benchmarks;
  PR results are posted back to the PR.
- **`publish-to-pypi.yml`** - triggered on tagged releases.
- **`pre-commit-ci.yml`** - automated pre-commit hook checks for format/lint/mypy.

