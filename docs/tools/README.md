# Docs tools

Developer-only helpers for maintaining the API surface image map.

## Update hotspots from Excalidraw

Run from repo root:

```bash
python docs/tools/update_api_surface_coords_from_excalidraw.py
```

This reads:

- `docs/tools/data/API_Surface_Feb_12.excalidraw`

And updates hotspot coordinates in:

- `docs/index.rst`

## Validate hotspot map

```bash
python docs/tools/validate_api_surface_map.py
```

This checks:

- rectangle coordinate validity (`x1 < x2`, `y1 < y2`)
- no negative coordinates
- bounds fit within configured base dimensions
- `alt` text is present
- links end with `.html`

## Optional local preview rebuild

```bash
python -m sphinx -b html -D exclude_patterns='**/*.ipynb' docs docs/_build/html-no-nb
```

Then open:

- `docs/_build/html-no-nb/index.html?mapdebug=1`
