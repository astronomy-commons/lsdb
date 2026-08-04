"""Regenerate docs/_static/API_Surface_Feb_12.png from its .excalidraw source.

Run from the repo root: ``python docs/tools/regen_api_surface_png.py``
then ``python docs/tools/check_api_surface.py``. See README.md for the node setup.

Renders via export_excalidraw.mjs (headless Excalidraw), then quantizes and asserts
the invariants that are easy to get wrong by hand:

* **exact size** -- the hotspot coords in docs/index.rst are hand-tuned PNG pixels, so
  a differently-sized export silently misaligns all of them. Size is
  ``trunc((bbox + 2*PADDING) * SCALE)``, hence the fixed padding/scale below.
* **palette mode** -- a straight RGB export is ~846 KB versus ~400 KB quantized.
  ``check-added-large-files`` only inspects *added* files, so overwriting in place
  would slip past the 500 KB hook and silently bloat the repo.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "data/API_Surface_Feb_12.excalidraw"
PNG = HERE.parent / "_static/API_Surface_Feb_12.png"
SCALE, PADDING = 3, 10
EXPECTED_SIZE = (3719, 2164)
MAX_KB = 500


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tmp) / "raw.png"
        cmd = [
            "node",
            str(HERE / "export_excalidraw.mjs"),
            str(SCENE),
            str(raw),
            "--scale",
            str(SCALE),
            "--padding",
            str(PADDING),
        ]
        if subprocess.run(cmd, check=False).returncode or not raw.is_file():
            print("export failed; see docs/tools/README.md for the node setup", file=sys.stderr)
            return 1
        size = Image.open(raw).size
        if size != EXPECTED_SIZE:
            print(
                f"REFUSING TO WRITE: exported {size[0]}x{size[1]}, expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}.\n"
                "The hotspots are hand-tuned to that size, so an element must have moved the "
                "scene's bounding box. Fix the drawing, or re-place every hotspot by hand.",
                file=sys.stderr,
            )
            return 1
        Image.open(raw).convert("RGB").quantize(colors=256).save(PNG, optimize=True)

    out, kb = Image.open(PNG), PNG.stat().st_size / 1024
    print(f"wrote {PNG}: {out.size[0]}x{out.size[1]} mode={out.mode} {kb:.0f} KB")
    if out.mode != "P" or kb > MAX_KB:
        print(f"expected palette mode under {MAX_KB} KB, got {out.mode} at {kb:.0f} KB", file=sys.stderr)
        return 1
    print("now run: python docs/tools/check_api_surface.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
