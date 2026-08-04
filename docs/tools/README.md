# Docs tools

Developer-only helpers for maintaining the clickable **API surface map** on the docs landing page.

That map is three coupled artifacts, and they must stay in sync:

| Artifact | Role |
| --- | --- |
| `data/API_Surface_Feb_12.excalidraw` | source of truth for the drawing |
| `../_static/API_Surface_Feb_12.png` | the rendered raster the page displays (3719×2164, palette mode) |
| `../index.rst` | the `<img>` plus 62 `<area>` hotspots whose coords are in **PNG-pixel space** |

Nothing in the normal build validates the hotspots: they live inside a `.. raw:: html` block, so
neither Sphinx nor `linkcheck` looks at them, and there is no docs job in CI. The two audit
scripts below are the only automated protection — **run them after any change to the figure.**

## Editing the figure

1. Edit `data/API_Surface_Feb_12.excalidraw` — either on <https://excalidraw.com> (Open, edit,
   Save to…) or programmatically.
2. Re-render the PNG: `python regen_api_surface_png.py` (from the repo root:
   `python docs/tools/regen_api_surface_png.py`).
3. If any label you touched has a hotspot, hand-edit its `<area>` in `../index.rst`.
4. Run both audits (below). Both must be clean.

### Keep the export at 3719×2164

Excalidraw exports at `trunc((bbox + 2 × padding) × scale)`, and this figure was produced at
**padding 10, scale 3** — `regen_api_surface_png.py` defaults to exactly that and *refuses to
write* on any other size, because the hotspot coords are hand-tuned to this raster.

So the drawing's **bounding box must not change**: `x[109.454, 1327.960] y[67.180, 768.715]`.
Editing text inside a box is safe; moving or resizing an extreme element is not. The four extremes
are the red legend rectangle (min x), `bg-core` (min y), `bg-io` (max x), and the text element
`' write_catalog'` (max y). That last one is a trap — it has `lineHeight` 4.29, so adding a single
line to it grows the export by ~342 px.

Two more things that bite when editing the JSON by hand:

- `text` and `originalText` must be updated **together**; tooling reads `originalText` first.
- Excalidraw does **not** re-measure text when loading a file, so a stale `width`/`height` is
  silently trusted. `height` is exactly `nLines × fontSize × lineHeight`. For `width`, measure with
  the real webfont and scale the stored value by the old→new ratio — plain `measureText` carries a
  ~2.7% systematic bias against the metrics that produced the committed values.
- Deleting a line of text also means adjusting the highlight rectangle behind it, or you leave a
  visible empty band. **Always look at the rendered PNG**; no audit catches cosmetic gaps.

## Regenerate the PNG

```bash
python docs/tools/regen_api_surface_png.py
```

Renders via `export_excalidraw.mjs` in headless Chromium — using Excalidraw's own `exportToBlob`,
so the bundled Excalifont/Virgil webfonts and the rough.js stroke jitter match the committed
asset — then quantizes to a 256-colour palette and asserts size, mode, and byte ceiling.

The quantization is not cosmetic: a straight RGB export of this drawing is ~846 KB versus ~400 KB
quantized. `check-added-large-files` (500 KB) only inspects *added* files, so overwriting the PNG
in place slips past the hook — but renaming the asset would make it fire.

### One-time setup

Node is required, and the export needs a browser. Install outside the repo:

```bash
export EXCALIDRAW_EXPORT_HOME=~/excalidraw-export   # anywhere outside the repo
mkdir -p "$EXCALIDRAW_EXPORT_HOME" && cd "$EXCALIDRAW_EXPORT_HOME"
npm init -y
npm install puppeteer @excalidraw/excalidraw react react-dom
npm install --no-save esbuild

# Build the browser bundle. It only needs to re-export Excalidraw's own helpers,
# so esbuild reads the entry from stdin -- no extra file in the repo.
# NODE_PATH is required so esbuild resolves @excalidraw/excalidraw from here
# (esbuild's CLI has no --node-paths flag; it reads NODE_PATH from the env).
echo 'import { exportToBlob, exportToCanvas } from "@excalidraw/excalidraw";
window.ExcalidrawLib = { exportToBlob, exportToCanvas };
window.__excalidrawReady = true;' \
  | NODE_PATH="$EXCALIDRAW_EXPORT_HOME/node_modules" \
    "$EXCALIDRAW_EXPORT_HOME/node_modules/.bin/esbuild" --bundle --format=iife --loader=js \
      --outfile="$EXCALIDRAW_EXPORT_HOME/bundle.js" \
      --define:process.env.NODE_ENV='"production"' --loader:.woff2=dataurl --minify
```

`export_excalidraw.mjs` finds this via `EXCALIDRAW_EXPORT_HOME` (default: `./excalidraw-export`
beside the script). Costs ~900 MB, so keep it off any quota-limited home directory — puppeteer
caches Chromium in `~/.cache/puppeteer` unless you set `PUPPETEER_CACHE_DIR`.

Use **puppeteer**, not playwright: puppeteer fetches Chromium from `storage.googleapis.com`,
whereas `cdn.playwright.dev` is blocked on some networks (SLAC USDF among them). Webfonts load
lazily, and an unloaded font silently falls back and shifts every glyph, so the exporter refuses
to write unless Excalifont actually loaded.

## Audits

```bash
python docs/tools/check_api_surface.py
```

It runs three fatal checks: **parse** (every `<area>` matches the expected attribute order — an
unparsed tag would be skipped by everything else), **links**, and **alignment**.

The **links** audit resolves each `href` against the pages Sphinx will actually generate.
`docs/reference/api/` is gitignored, so the authority is the `autosummary` blocks carrying
`:toctree: api` in the curated `docs/reference/*.rst`; the script self-checks by confirming it
reproduces the on-disk stub set exactly. Note the class-level `autosummary` blocks in
`docs/reference/api/lsdb.catalog.Catalog.rst` have **no** `:toctree:`, so a name appearing there is
*not* evidence its page exists.

The **alignment** audit samples PNG ink under each hotspot and compares the ink centroid
to the rectangle centre. It separates two failure classes:

- **structural** — the hotspot is out of bounds or sits on blank canvas, i.e. its label moved or was
  deleted. Always fails; `EXPECTED_FLAGS` never excuses this.
- **off-centre** — ink is present but not centred. Fails only for names outside `EXPECTED_FLAGS`,
  which lists the eight labels whose boxes were widened by hand to span a whole line.

Judge the off-centre set by **membership, not absolute numbers**. Note the ink threshold
(`INK_MAX_LUMA = 210`) has to accommodate *coloured* labels — the orange stream labels sit at
luminance ~154, while the pale highlight fills are all ≥234.

It verifies that ink is present under a hotspot, not that the ink is the *right* label (that would
need OCR), so it will not catch a hotspot that lands on a different adjacent label. Use the
`?mapdebug=1` preview below for that.

## Hotspot coordinates

Coords are PNG pixels. The transform from Excalidraw canvas coordinates is

```
x_png = (x_canvas - 109.454 + 10) * 3
y_png = (y_canvas -  67.180 + 10) * 3
```

To move a hotspot because its element moved, **shift the existing rectangle** by the element's
delta (`dy_canvas × 3`) rather than recomputing it from scratch — the committed values contain
deliberate hand-tuning that a fresh computation throws away.

### Why there is no automated coordinate updater

There used to be one (`update_api_surface_coords_from_excalidraw.py`, removed 2026-08-04 — see
history if you want the code). It derived each hotspot by locating its label's token within a line
**by character index**, which assumes a monospace font. The drawing uses Excalifont, which is
proportional, so the arithmetic drifted. Measured against the unmodified source, it rewrote
**15 of 62 hotspots with no input change and no warning** — `MarginCatalog` by 138 px,
`write_catalog` by 115 px, `tail` by 144 px — and matched the wrong element outright for
`MarginCatalog` and `write_catalog`. Its fuzzy matching (`min_overlap = 0.60`) also made a stale
`alt` dangerous rather than loud: left in place, `alt="merge"` snapped silently onto
**`merge_map`'s** box.

Because it could not reproduce the committed coordinates, those coordinates are hand-tuned, and the
script was strictly a liability. If you want to rebuild it, measure **real Excalifont advance
widths** — the headless browser above makes that practical — and check it against
`check_api_surface.py` before trusting a single coordinate.

## Optional local preview

```bash
python -m sphinx -b html -D exclude_patterns='**/*.ipynb' docs docs/_build/html-no-nb
```

Then open `docs/_build/html-no-nb/index.html?mapdebug=1` — the `mapdebug` flag draws the hitboxes
over the image, which is the definitive visual check. Hard-reload; the PNG caches aggressively.
(`pandoc` is required for the notebook-inclusive build, hence `exclude_patterns` here.)
