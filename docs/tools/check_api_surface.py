"""Audit the clickable API-surface image map on the docs landing page.

Run from the repo root: ``python docs/tools/check_api_surface.py``

Nothing else checks this map -- it lives in a ``.. raw:: html`` block, so the Sphinx
build and ``linkcheck`` both ignore it. See README.md for the full rationale.

Three checks, all fatal:
  parse      every ``<area>`` tag matches AREA_RE (catches missing/malformed coords,
             missing href/alt, empty alt, reordered attributes)
  links      every href resolves to a page Sphinx will generate
  alignment  every hotspot sits on ink, and only baselined boxes are off-centre
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
from PIL import Image

DOCS = pathlib.Path("docs")
PNG = DOCS / "_static/API_Surface_Feb_12.png"
AREA_RE = r'<area shape="rect" coords="([0-9]+,[0-9]+,[0-9]+,[0-9]+)"\s+href="([^"]+)"\s+alt="([^"]+)"'

# Ink threshold must catch COLOURED labels: the orange stream labels sit at luminance
# ~154, so a tighter cut reports "no ink" for them. Pale highlight fills are all >=234.
INK_MAX_LUMA = 210

# Boxes hand-widened to span a whole line, so their ink centroid legitimately sits
# off-centre. Measured against main's figure (a9492a9a) with this same detector.
EXPECTED_OFF_CENTRE = {
    "concat",
    "crossmatch",
    "join",
    "name",
    "nested_columns",
    "show_versions",
    "tail",
    "to_delayed",
}


def curated_stubs() -> set[str]:
    """Stub basenames the curated reference pages will generate.

    ``docs/reference/api/`` is gitignored, so the authority is the ``autosummary``
    blocks carrying ``:toctree: api``. Blocks without it generate nothing.
    """
    stubs: set[str] = set()
    for rst in sorted(DOCS.glob("reference/*.rst")):
        module = ""
        lines = rst.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            if mod := re.match(r"\s*\.\.\s+currentmodule::\s*(\S+)", lines[i]):
                module = mod.group(1)
            elif re.match(r"\s*\.\.\s+autosummary::", lines[i]):
                i += 1
                toctree = False
                while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith(":")):
                    toctree = toctree or "toctree" in lines[i]
                    i += 1
                while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                    entry = lines[i].strip()
                    if entry and toctree and not entry.startswith((":", "..")):
                        stubs.add(f"{module}.{entry}" if module else entry)
                    i += 1
                continue
            i += 1
    return stubs


def main() -> int:
    txt = (DOCS / "index.rst").read_text(encoding="utf-8")
    found = [
        (m.group(3), m.group(2), tuple(int(v) for v in m.group(1).split(",")))
        for m in re.finditer(AREA_RE, txt)
    ]
    problems: list[str] = []

    # parse -- an unparsed tag would be silently skipped by every check below
    declared = txt.count("<area")
    print(f"hotspots: {len(found)} parsed of {declared} <area> tags")
    if not found:
        problems.append("no <area> tags parsed at all")
    if declared != len(found):
        problems.append(
            f"{declared - len(found)} <area> tag(s) did not parse -- check attribute order/spelling"
        )

    # links
    stubs = curated_stubs()
    on_disk = {p.stem for p in DOCS.glob("reference/api/*.rst")}
    if on_disk and on_disk != stubs:
        # curated_stubs() no longer models Sphinx, so the authority is untrustworthy
        # and a green result would be meaningless.
        for name in sorted(on_disk ^ stubs):
            print(f"  STUB MISMATCH {name}")
        problems.append(f"stub cross-check disagreed (curated={len(stubs)} on_disk={len(on_disk)})")
    for alt, href, _ in found:
        target = href.split("#")[0].removesuffix(".html")
        ok = (
            target.removeprefix("reference/api/") in stubs
            if target.startswith("reference/api/")
            else (DOCS / f"{target}.rst").is_file()
        )
        if not ok:
            problems.append(f"dead href: {alt} -> {href}")

    # alignment
    grey = np.asarray(Image.open(PNG).convert("L")).astype(int)
    height, width = grey.shape
    off_centre = set()
    for alt, _, (x0, y0, x1, y1) in found:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            problems.append(f"out of bounds: {alt} ({x0},{y0},{x1},{y1})")
            continue
        ink = grey[y0:y1, x0:x1] < INK_MAX_LUMA
        if not ink.any():
            # A hotspot on blank canvas means its label moved or was deleted. This is
            # structural and is never excused by EXPECTED_OFF_CENTRE.
            problems.append(f"no ink under hotspot: {alt}")
            continue
        ys, xs = np.nonzero(ink)
        dx = round(xs.mean() - ink.shape[1] / 2)
        dy = round(ys.mean() - ink.shape[0] / 2)
        if ink.mean() * 100 < 2.0 or abs(dx) > 45 or abs(dy) > 16:
            off_centre.add(alt)
    if new := off_centre - EXPECTED_OFF_CENTRE:
        problems.append(f"newly off-centre: {sorted(new)}")
    if stale := EXPECTED_OFF_CENTRE - off_centre:
        print(f"  NOTE no longer off-centre, tighten EXPECTED_OFF_CENTRE: {sorted(stale)}")

    if problems:
        print(f"\nFAIL ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("OK -- parsed, all hrefs resolve, all hotspots on their labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
