# Notebook style guide — pilot run

*Working document for the pilot PR. Not part of the rendered docs; delete or
archive once the full revision pass lands.*

## What this PR does

`docs/developer/notebook_style_guide.md` was added in `283c9c61` and had never been applied to
anything. Rather than batch-editing all 43 tutorial notebooks against an untested guide, this PR
runs it against a small, deliberately varied subset and reports back on where the guide itself was
wrong.

The headline result is in the next section: **six guide/template rules were wrong or ambiguous, and
measuring the notebooks first turned what would have been edits to 30+ notebooks into edits to the
guide.** That is the argument for piloting.

## Changes to the guide and template (what the pilot bought us)

Every change here came from measuring what the notebooks already do, then finding the guide in the
minority. In each case the guide was changed, not the notebooks.

### 1. Section numbering: `## 1 - Title` → `## 1. Title`

The guide specified a dash. The notebooks overwhelmingly use a period:

| Form | Occurrences across all 43 notebooks |
|---|---|
| `## 1. Title` | **120** |
| `## 1 - Title` | 3 (all in `dask_client.ipynb`) |

Following the guide literally would have meant editing **32 notebooks** to change punctuation, in
the direction of a convention exactly one file uses. Changing the guide instead reduces the eventual
numbering work to `dask_client.ipynb` alone.

### 2. Sub-section numbering: `### 1.1 - Title` → `### 1.1 Title`

Here there was no majority to defer to, so this one was an explicit editorial call rather than a
measurement:

| Form | Occurrences |
|---|---|
| `### 1.1 Title` | 29 |
| `### 1.1. Title` | 28 |
| `### 1.1 - Title` | 4 (the guide's form) |

Chosen: no trailing dot, which avoids the awkward `1.1.` double period. This leaves 28 notebooks
with a trailing dot to normalize during the full pass — the one place where the guide is
deliberately asking for churn, with eyes open.

The guide now spells the two levels out explicitly, since they differ:

> Content sections, numbered (`## 1. ...`, with `### 1.1 ...` sub-sections as needed). Note the
> punctuation: a period after a top-level number, nothing after a sub-section number.

### 3. Objectives line: `In this tutorial we will:` → `In this tutorial, we will:`

| Form | Occurrences |
|---|---|
| `In this tutorial, we will:` | **33** |
| `In this tutorial we will:` | 1 (`map_partitions.ipynb`) |

Same shape as the numbering case, smaller stakes. Changed in both the structure list and the
checklist.

### 4. Template: unnumbered sections ahead of `## 1.`

The guide states that the template is its executable counterpart and that "where the two disagree,
one of them is wrong." They disagreed. `tutorial_template.ipynb` placed two content sections —
`## Open a catalog` and `## Plotting lightcurves` — *unnumbered*, ahead of the numbered
`## 1. [Interesting thing 1]` placeholder. Anyone starting from the template produced a notebook
that fails the guide's own checklist.

The template's sections are now a single continuous sequence:

```
## Introduction
## 1. Open a catalog
## 2. Plotting lightcurves
## 3. [Interesting thing 1]
## 4. [Interesting thing 2]
###   4.1 [Sub-section 1, if applicable]
###   4.2 [Sub-section 2, if applicable]
## Close the Dask client
## About
```

This was a template bug, not a guide bug — the guide's stated section order was right all along.

### 5. `## About` field labels: `**Author(s):**` → `**Authors**:`

The template was in the minority on two axes at once:

| Axis | Majority | Template had |
|---|---|---|
| Label | `Authors` (34) | `Author(s)` (6) |
| Colon on the author line | outside the bold (33) | inside (7) |
| Colon on the date line | outside the bold (27) | inside (14) |

Template updated on all three; the guide now names the labels explicitly rather than leaving them
to be inferred from the template. **Deliberately not pinned: the date format.** `January 19, 2026`
(22), `Jan 19, 2026` (15), and `19 May 2026` (3) coexist with no clear winner, and normalizing them
would mean touching 40 notebooks for no reader benefit. The guide says so out loud, so the next
person doesn't re-open it.

### 6. Imports placement — a descriptive rule instead of a prescriptive one

The guide said only "import cells follow [the intro]," which was ambiguous enough to be unusable as
a checklist item. Practice is genuinely split:

| Placement | Count |
|---|---|
| Inside/after the first numbered section | 20 |
| Before the first numbered section | 11 |
| No numbered sections at all | 10 |

Rather than pick a side and move imports by one or two cells across 20-odd notebooks for no
functional gain, the guide now states the part that actually matters as a rule — **imports come no
earlier than the Introduction**, so a reader is never hit with code before they know what the
notebook is for — and the rest as a stated preference. Both existing placements are compliant.

### Not changed, and why

- **Logging level and Dask dashboard printing.** Frozen at the author's request pending a decision
  on whether to keep those rules at all. Current state is inventoried below so that decision can be
  made from data. Nothing was applied, removed, or reformatted.
- **The Plotting section.** Untested this round — see the constraint below.

## Constraints on this pilot

**No re-execution is available.** Notebooks in `docs/tutorials/pre_executed/` ship with saved outputs
and `nbsphinx.execute: "never"`. Editing a code cell there desyncs the code from the rendered output:
the published page would show the old figure beside new code. So **this pilot makes no code-cell
edits in `pre_executed/`**.

That has a significant consequence: **the entire Plotting section of the guide goes untested.** Every
matplotlib plot in the docs lives in `pre_executed/`. The 10 executed notebooks contain no matplotlib
plots at all — `region_selection.ipynb`'s plots are LSDB `plot_pixels` sky maps, which the guide's
rules (inverted magnitude axes, error bars, hexbin, colormaps) do not govern. Validating the plotting
rules requires a follow-up pass by someone who can re-run these notebooks against real data.

## Survey: all 43 notebooks

Executed = `docs/tutorials/*.ipynb`, run at build time by nbsphinx; outputs stripped on commit by the
`jupyter-nb-clear-output` hook. Pre-exec = `docs/tutorials/pre_executed/*.ipynb`, outputs preserved,
`nbsphinx.execute: "never"`, enforced by the `pre-executed-nb-never-execute` hook.

### Structure and footer

| Notebook | Group | Numbering | Objectives | `## Introduction` | `## About` author | Last updated |
|---|---|---|---|---|---|---|
| catalog_object | exec | `1.` | yes | yes | Sandro Campos, Melissa DeLucchi, and Sean McGuire | Jan 19, 2026 |
| column_filtering | exec | `1.` | yes | yes | Olivia Lynn | May 20, 2025 |
| dask_client | exec | **`1 -`** | yes | yes | Olivia Lynn and Melissa DeLucchi | May 22, 2025 |
| exporting_results | exec | none | **missing** | **missing** | Neven Caplar, Sandro Campos | April 27, 2025 |
| import_catalogs | exec | `1.` | yes | yes | Sandro Campos | April 4, 2025 |
| lazy_operations | exec | **none** | yes | yes | Sean McGuire | June 27, 2025 |
| margins | exec | `1.` | yes | yes | Sean McGuire | April 18, 2025 |
| region_selection | exec | `1.` | yes | yes | Sandro Campos and Melissa DeLucchi | August 29, 2025 |
| row_filtering | exec | `1.` | yes | yes | Sandro Campos, Melissa DeLucchi, Olivia Lynn, and Derek Jones | April 14, 2025 |
| small_scale | exec | `1.` | "we will cover" | yes | Olivia Lynn | May 18, 2026 |
| access.pyvo | pre-exec | `1.` | yes | yes | Melissa DeLucchi | May 14, 2026 |
| crossmatching | pre-exec | `1.` | yes | yes | Derek Jones | Oct 27, 2025 |
| custom_search | pre-exec | `1.` | yes | yes | Melissa DeLucchi | October 27, 2025 |
| des-gaia | pre-exec | `1.` | yes | yes | Konstantin Malanchev | Oct 27, 2025 |
| dp1-gaia-epoch-prop | pre-exec | `1.` | **missing** | **missing** | **no About at all** | — |
| explode_lightcurves | pre-exec | none | yes | **missing** | **no About at all** | — |
| full_dp2_crossmatches | pre-exec | **none** | "In this notebook we'll" | **missing** | Doug Branton | July 27, 2026 |
| index_table | pre-exec | `1.` | yes | yes | Melissa DeLucchi | October 27, 2025 |
| join_catalogs | pre-exec | `1.` | yes | yes | Sandro Campos | Oct 27, 2025 |
| manual_verification | pre-exec | `1.` | **missing** | **missing** | Melissa DeLucchi | Oct 27, 2025 |
| map_partitions | pre-exec | `1.` | **no comma** | yes | Derek Jones | January 19, 2026 |
| nestedframe | pre-exec | `1.` | yes | yes | Doug Branton | October 27, 2025 |
| plotting | pre-exec | `1.` | yes | yes | Sean McGuire | October 27, 2025 |
| rubin_dp1 | pre-exec | `1.` | yes | yes | Neven Caplar, Derek Jones, Konstantin Malanchev, Olivia Lynn | May 13, 2026 |
| rubin_dp1_photoz | pre-exec | `1.` | yes | **missing** | Sandro Campos, Sarah Pelesky, Tianqing Zhang | Jan 29, 2026 |
| rubin_dp1_vsx | pre-exec | `1.` | yes | yes | Konstantin Malanchev | May 12, 2026 |
| rubin_dp2-da-white-dwarfs | pre-exec | `1.` | yes | yes | Konstantin Malanchev | July 27, 2026 |
| rubin_dp2 | pre-exec | `1.` | yes | yes | Neven Caplar, Sandro Campos, Olivia Lynn, Konstantin Malanchev | July 27, 2026 |
| rubin_dp2_historical_dia | pre-exec | **none** | **missing** | **missing** | Sandro Campos, Konstantin Malanchev, Neven Caplar | July 27, 2026 |
| rubin_dp2_host_galaxies | pre-exec | **none** | **missing** | **missing** | Sandro Campos, Konstantin Malanchev, Neven Caplar | July 27, 2026 |
| rubin_dp2_photoz | pre-exec | `1.` | yes | **missing** | Sandro Campos, Sarah Pelesky, Tianqing Zhang, Konstantin Malanchev | July 27, 2026 |
| rubin_dp2_starter | pre-exec | **none** | **missing** | **missing** | Heather Sestili | July 27, 2026 |
| rubin_dp2_tutorial | pre-exec | **none** | yes | **missing** | Heather Sestili | July 27, 2026 |
| rubin_dp2_why_lazy_evaluation | pre-exec | **none** | **missing** | **missing** | Heather Sestili | July 27, 2026 |
| scaling_workflows | pre-exec | **none** | yes | yes | Doug Branton | July 17, 2025 |
| timeseries | pre-exec | `1.` | yes | yes | Konstantin Malanchev, Derek Jones | Oct 27, 2025 |
| types_of_crossmatch | pre-exec | `1.` | yes | **missing** | Sean McGuire | 19 May 2026 |
| using_rubin_data | pre-exec | `1.` | yes | yes | Doug Branton | 11 May 2026 |
| visualize_periodic_lcs | pre-exec | **none** | yes | yes | Sandro Campos, Doug Branton | 7 Jan 2026 |
| ztf-alerts-sne | pre-exec | `1.` | **missing** | **missing** | Konstantin Malanchev and Mi Dai | April 17, 2025 |
| ztf_bts-ngc | pre-exec | `1.` | "In this notebook, we demonstrate" | **missing** | Konstantin Malanchev, Mi Dai | Oct 27, 2025 |
| zubercal-ps1-snad | pre-exec | `1.` | **missing** | **missing** | Konstantin Malanchev | Oct 27, 2025 |

Totals: numbering `1.` = 32, `1 -` = 1, none = 10 · objectives present = 35 · Introduction present =
26 · **`## About` present = 40 of 42, and where present, author and date are filled in every one.**

**The `## About` footer is in far better shape than a first pass suggested.** Only
`dp1-gaia-epoch-prop.ipynb` and `explode_lightcurves.ipynb` lack the section entirely; those two are
the whole of the work. What the footers need is not authorship research but *format* normalization
— see the label and date-format inconsistencies below.

### `## About` field formatting

Three separate inconsistencies, none of which the guide currently rules on. In each case the
template is either silent or in the minority.

| Axis | Variants found | Template currently says |
|---|---|---|
| Label | `Authors` (34) · `Author(s)` (6) | `Author(s)` — **minority** |
| Colon | `**Authors**:` outside the bold (33) · `**Authors:**` inside (7) | inside — **minority** |
| Date label | `Last updated on` (23) · `Last run` (10) · `Last updated/verified on` (3) · `Last updated /verified on` (4) | `Last updated on` — majority ✓ |
| Date format | `January 19, 2026` (22) · `Jan 19, 2026` (15) · `19 May 2026` (3) | `[Date]` — unspecified |

Note that `Last run` (10 notebooks, all `pre_executed/`) is arguably the more accurate label for that
group: a pre-executed notebook's outputs were produced on a specific date, which is a different fact
from when someone last edited the prose. Worth deciding deliberately rather than flattening.

### Dask logging / dashboard / client (frozen this round — inventory only)

| Notebook | Logging calls | Dashboard link | Client form |
|---|---|---|---|
| full_dp2_crossmatches | root=WARNING; distributed=WARNING | prints | `with Client` |
| rubin_dp2-da-white-dwarfs | root=WARNING; distributed=WARNING | prints | `with Client` |
| rubin_dp2_host_galaxies | root=WARNING; distributed=WARNING | prints | `with Client` |
| rubin_dp2_photoz | root=WARNING; distributed=WARNING | prints | `with Client` |
| rubin_dp2_starter | root=WARNING; distributed=WARNING | prints | `client = Client(...)` |
| rubin_dp2_tutorial | root=WARNING; distributed=WARNING | prints | `client = Client(...)` |
| rubin_dp2_why_lazy_evaluation | root=WARNING; distributed=WARNING | prints | `client = Client(...)` |
| rubin_dp2_historical_dia | root=WARNING; distributed=WARNING | — | no client |
| rubin_dp2 | — | **prints** | `client = Client(...)` |
| column_filtering, dask_client, lazy_operations, region_selection, row_filtering, small_scale | — | — | `client = Client(...)`, closed |
| crossmatching, des-gaia, map_partitions, plotting, timeseries | — | — | `client = Client(...)`, closed |
| **rubin_dp1** | — | — | `client = Client(...)`, **never closed** |
| **rubin_dp2** | — | — | `client = Client(...)`, **never closed** |
| import_catalogs, dp1-gaia-epoch-prop, rubin_dp1_vsx, scaling_workflows, visualize_periodic_lcs, ztf-alerts-sne, ztf_bts-ngc | — | — | `with Client` (already compliant) |
| access.pyvo, catalog_object, custom_search, exporting_results, explode_lightcurves, index_table, join_catalogs, manual_verification, margins, nestedframe, rubin_dp1_photoz, types_of_crossmatch, using_rubin_data, zubercal-ps1-snad | — | — | no client |

Two observations for the pending logging/dashboard decision:

- **"Close what you open" is nearly satisfied already.** Only `rubin_dp1.ipynb` and `rubin_dp2.ipynb`
  hold a persistent client and never close it. Everything else either uses `with Client(...)` — which
  the guide should probably name as the preferred form — or already calls `client.close()`.
- **The dashboard rule means something different in the two groups.** All 8 dashboard-printing
  notebooks are `pre_executed/`, so their link was captured once on RSP and never re-runs. Applying
  that rule to the 10 executed notebooks is a different proposition: those run on ReadTheDocs, where
  `dashboard_link` resolves against the RTD build container and would bake a dead URL into the
  published page.

## Pilot subset

Five notebooks, chosen to span every axis that could make the guide behave differently — not to be
representative.

| # | Notebook | Why it's in the pilot |
|---|---|---|
| 1 | `tutorials/lazy_operations.ipynb` | Executed, **no numbering at all** → tests inventing a scheme from scratch. Build-verifiable end to end. |
| 2 | `tutorials/exporting_results.ipynb` | Executed, 2 cells, no objectives / Introduction / numbering → the "almost nothing to work with" case. |
| 3 | `tutorials/pre_executed/types_of_crossmatch.ipynb` | Pre-exec, already numbered, missing Introduction, single author → the **cheap markdown-only pass**. Sets the floor for diff size. |
| 4 | `tutorials/pre_executed/ztf_bts-ngc.ipynb` | Pre-exec, **hyphenated filename** (rename + toctree update), non-standard objectives phrasing, and **has plots we deliberately leave alone** → shows what a structure-only pass leaves visibly unfinished. |
| 5 | `tutorials/pre_executed/rubin_dp2_starter.ipynb` | Pre-exec, **879 KB**, no numbering / objectives / Introduction, and **already has logging + dashboard** → worst-case diff noise, and proves the freeze held. |

### Applied

| Notebook | Diff (+/−) | What changed |
|---|---|---|
| `lazy_operations.ipynb` | 4 / 4 | Numbered 2 sections + 1 sub-section; promoted `### Closing the Dask client` to a top-level `## Close the Dask client` |
| `exporting_results.ipynb` | 33 / 8 | Split 1 prose cell into 5: added objectives and `## Introduction`, split the body into 3 numbered sections |
| `types_of_crossmatch.ipynb` | 12 / 1 | Added `## Introduction`; `**Author(s)**:` → `**Authors**:` |
| `ztf_bts_ngc.ipynb` | 18 / 6 | Renamed from `ztf_bts-ngc.ipynb`; rewrote objectives into the standard form; added `## Introduction`; dropped trailing dots from `### 1.1.` / `### 1.2.` |
| `rubin_dp2_starter.ipynb` | 37 / 7 | Added objectives and `## Introduction`; numbered 6 sections; added a `## Close the Dask client` heading for the existing `client.close()` cell; colon moved outside the bold on `**Last run**:` |

**Verified: all three `pre_executed/` notebooks have byte-identical code cells *and* byte-identical
outputs** before and after. Nothing desynced.

### Deliberately not applied

- Any code-cell edit in `pre_executed/` (no re-execution available)
- The Plotting section in full (all plots live in `pre_executed/`)
- Logging level and dashboard printing (frozen)
- The other 6 hyphenated filenames — `des-gaia`, `ztf-alerts-sne`, `zubercal-ps1-snad`,
  `rubin_dp2-da-white-dwarfs`, `dp1-gaia-epoch-prop`, `access.pyvo`. Renaming changes published
  `docs.lsdb.io` URLs, so it belongs in one deliberate PR rather than as a side effect of this one.

## What applying it actually taught us

Four things surfaced only once the guide was used in anger, rather than read.

**The "output-heavy notebooks will produce unreviewable diffs" worry was wrong.** That was the whole
reason `rubin_dp2_starter.ipynb` (879 KB) was in the pilot. A markdown-only pass on it produced a
**37-line insertion / 7-line deletion** diff — entirely reviewable. Because nbformat stores each
output blob as a single long JSON line, output cells simply don't move when markdown cells around
them change. **This removes the main objection to batch-revising the `pre_executed/` group.** The
constraint that matters there is code-cell edits, not file size.

**Renaming needs a repo-wide grep, not a `tutorial_toc/` grep.** The rename of `ztf_bts-ngc.ipynb`
had a second inbound reference in `docs/data-access/vizier.rst`, outside the toctree directory
entirely. A `:doc:` reference to a renamed notebook is a hard Sphinx error, so the eventual 6-file
rename PR must grep all of `docs/`, not just the toc files.

**The guide doesn't say where `## Close the Dask client` goes.** In `lazy_operations.ipynb` it
existed as `### Closing the Dask client`, nested inside a content section. The template puts it at
top level and unnumbered, which is what we did — but the guide should say so, and should pin the
heading text, since "Close" vs "Closing" already varies.

**The guide has nothing to say about `raw` cells.** `types_of_crossmatch.ipynb` uses `raw` cells for
`.. nbinfo::` admonitions, interleaved with the markdown structure. They aren't prose, aren't code,
and the section-order rule doesn't obviously apply to them. Worth a sentence.

### Judgment calls left alone, flagged instead of fixed

- **`## 4. Make some plot`** in `ztf_bts_ngc.ipynb` — a grammar slip in a heading. The guide has no
  copyedit rule, and inventing one mid-pass would make the revision unbounded. Decide whether a
  revision pass is allowed to fix obvious typos.
- **`**Last run**:` in `rubin_dp2_starter.ipynb`** — normalized the colon but kept `Last run` rather
  than forcing `Last updated on`, since the pilot doc flags that as an open question for
  `pre_executed/` notebooks. Preempting it here would have decided it by accident.
- **`## Introduction - What are Lazy Operations?`** — kept the descriptive suffix rather than
  flattening to a bare `## Introduction`. The guide's intent is satisfied and the suffix carries
  information.

## Verification

1. `pre-commit run --all-files` — confirms outputs stripped from the two executed notebooks, and that
   the three pre-executed ones kept `nbsphinx.execute: "never"`.
2. `cd docs && make no-nb` — structural build; catches a toctree break if the rename and
   `toc_science.rst` fall out of sync.
3. `cd docs && make html` — full build. Executes the two `tutorials/` notebooks for real; needs
   network access for the catalogs they open.
4. Spot-check the three pre-executed pages in `_readthedocs/html/`: no figure should now sit next to
   code that did not produce it.
5. `git diff --stat` on `rubin_dp2_starter.ipynb` — a large diff from a markdown-only pass on an
   879 KB file is itself a finding about batch-revising this group.

## Follow-up work this pilot identifies

- **Decide the logging / dashboard rules**, using the inventory above. The RTD-vs-RSP split is the
  crux.
- **Add an `## About` section to the 2 notebooks missing one** — `dp1-gaia-epoch-prop.ipynb` and
  `explode_lightcurves.ipynb`. These are the only two needing human authorship input.
- **Normalize `## About` field formatting** across 42 notebooks — see the inconsistencies below.
- **Validate the Plotting section** — requires someone who can re-execute `pre_executed/` notebooks.
- **Normalize 28 notebooks** from `### 1.1.` to `### 1.1`, and `dask_client.ipynb` from `## 1 -` to `## 1.`.
- **Rename the remaining 6 hyphenated notebooks** in a dedicated PR, accepting the URL breakage.
- **Consider naming `with Client(...)` as the preferred form** in the guide's "close what you open"
  rule — 7 notebooks already use it and it makes the rule unbreakable.
