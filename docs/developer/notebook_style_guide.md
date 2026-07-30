# Tutorial notebook style guide

These are the conventions we follow for notebooks under `docs/tutorials/` (including
`docs/tutorials/pre_executed/`). They are loose standards, not hard rules: the goal is that
notebooks written by different authors look like they belong to the same documentation set,
and that a batch revision pass across all of them is mechanical rather than a judgment call
on every plot.

Start from [`docs/developer/tutorial_template.ipynb`](tutorial_template.ipynb). The template is
the executable version of this page. Where the two disagree, one of them is wrong — fix it
rather than picking a side silently.

## Structure

Follow the section order in the template:

1. Title (`# [Notebook Title]`)
2. "In this tutorial we will:" — a short bulleted list of what the reader will learn
3. `## Introduction` — what the feature is and, briefly, why it exists
4. Content sections, numbered (`## 1 - ...`, with `### 1.1 - ...` sub-sections as needed)
5. `## Close the Dask client` — if a client was opened for the notebook
6. `## About` footer

**Prose comes before code.** The intro text is the first thing a reader sees, so they know what
the notebook is for before they hit any setup. Import cells follow it.

**Close what you open.** If a notebook instantiates a Dask client that lives for the whole
notebook, close it at the end with `client.close()`.

**Every notebook needs the footer.** The `## About` section carries author(s), the date the
notebook was last updated/run, and the citation reminder. Copy it verbatim from the template
and fill in the fields.

## Logging and the Dask dashboard

Dask emits a lot of `INFO`-level logging, which drowns out the actual output of a tutorial.
Raise the threshold at the bottom of the main import cell (or in a cell immediately after it):

```python
import logging

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("distributed").setLevel(logging.WARNING)
```

Suppressing `INFO` also suppresses the line where Dask announces its dashboard, so print the
link explicitly when the client is created. Put the `print` *above* the bare `client` line, so
the link appears before the client repr:

```python
from dask.distributed import Client

client = Client(n_workers=4, threads_per_worker=1, memory_limit="auto")
print(f"Dask dashboard: {client.dashboard_link}")
client
```

> **Platform note.** `dashboard_link` fills in a template string from environment variables,
> so it is only correct on the platform the notebook was written for. The snippet above is
> right on the Rubin Science Platform. Elsewhere — USDF, for instance — `distributed` will
> warn `Failed to format dashboard link, unknown value: 'JUPYTERHUB_PUBLIC_URL'` and hand back
> a URL that does not resolve.
>
> There is no portable version of this line. Setting
> `dask.config.set({"distributed.dashboard.link": ...})` only lets you hardcode the correct
> URL for a platform you already know, so it is a fix per platform, not a general one. Before
> committing, check that the link you print actually opens where you are running. If it does
> not, drop the `print` and explain in prose how to reach the dashboard there instead.

## Naming and terminology

**Filenames use underscores**, not hyphens: `catalog_crossmatch_demo.ipynb`, not
`catalog-crossmatch-demo.ipynb`.

**Say "Rubin", not "LSST"**, when naming a data release or data product — "Rubin DP2", not
"LSST DP2". This matches how Rubin names its own releases ("Rubin Data Preview 1", "Rubin
Science Platform"). Applies to prose, plot titles, axis labels, and section headings.

The survey itself is still the Legacy Survey of Space and Time (LSST), and instruments and
software keep the names they have — LSSTCam, LSST Science Pipelines, `lsst.sphgeom`. The rule
is about our data, not a global find-and-replace.

## Plotting

The template's lightcurve cell is the reference recipe. The rules below are what we want a
reader to get out of any plot in the docs; use judgment on the exact numbers, but err toward
"visible in a rendered doc page at half width" over matplotlib defaults.

**Magnitude axes are inverted.** Brighter objects have lower magnitudes and belong at the top
of the plot. `plt.gca().invert_yaxis()` — do this every time a magnitude is on an axis.

**Show error bars, with caps.** If the catalog has an error column for the quantity being
plotted, plot it. Use `capsize` so the bars are readable as error bars rather than as smudges;
thicken them (`elinewidth`) so they survive the rendered page.

**Size for legibility.** Axis labels, tick labels, and legends should be large; markers should
be big enough to see against their error bars. The matplotlib defaults are too small for
documentation. Label both axes, with units where they exist.

**Do not silently drop points.** If you cut the axis range to make the interesting region
visible, show the excluded points as downward-pointing arrows at the edge (`lolims`/`uplims`
on `errorbar`, or a marker like `matplotlib.markers.CARETDOWNBASE`) so the reader knows they
exist. A lightcurve with one or two outlying epochs is usually better served by a tight zoom
plus arrows than by a full-range plot where the real variation is a flat line.

**Dense scatter plots should not be scatter plots.** Above a few thousand points, overplotting
hides the structure. Use `hexbin` or a 2D histogram instead, and consider a log color scale
when the density spans orders of magnitude.

**Prefer high-contrast colormaps.** `magma` or `viridis` over the matplotlib default. Zoom the
axes to the region where the data actually lives rather than showing the full range.

## Checklist

For a new notebook, or a revision pass over an existing one:

- [ ] Filename uses underscores
- [ ] Title, then the "In this tutorial we will:" objectives, then the Introduction, then the imports
- [ ] Logging cell present, at the bottom of / just after the main imports
- [ ] Dashboard link printed above the bare `client` line, and the link works on the target platform
- [ ] "Rubin", not "LSST", when naming data releases and data products
- [ ] Plotting:
    - [ ] Magnitude axes inverted
    - [ ] Error bars present and capped where error columns exist
    - [ ] Labels and markers legible at documentation size
    - [ ] Points outside a cut shown as arrows, not dropped
    - [ ] Dense scatters plotted as hexbin/2D histogram
    - [ ] High-contrast colormap (`magma`, `viridis`) rather than the matplotlib default
- [ ] `## About` footer with author, last-updated date, citation link
- [ ] Any long-lived client is closed
