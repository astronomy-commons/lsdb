"""Update API-surface image-map coords from an Excalidraw source file.

This script is the canonical maintenance entrypoint for hotspot regeneration.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EXCALIDRAW_PATH = ROOT / "docs/tools/data/API_Surface_Feb_12.excalidraw"
DEFAULT_INDEX_PATH = ROOT / "docs/index.rst"


@dataclass(frozen=True)
class UpdateConfig:
    excalidraw_path: pathlib.Path
    index_path: pathlib.Path
    png_width: int = 3719
    png_height: int = 2164
    export_pad: int = 9


@dataclass(frozen=True)
class UpdateResult:
    updated_areas: int
    unmatched_alts: tuple[str, ...]
    transform_scale: float


# ---------------------------------------------------------------------------
# Coordinate transform: Excalidraw canvas units  ->  PNG pixels
# ---------------------------------------------------------------------------

def compute_transform(elements: list, *, png_width: int, png_height: int, export_pad: int) -> dict:
    """Return affine params {scale, offset_x, offset_y} mapping Excalidraw
    canvas coordinates to PNG-pixel coordinates."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for e in elements:
        if e.get("isDeleted"):
            continue
        x = float(e.get("x", 0))
        y = float(e.get("y", 0))
        w = float(e.get("width", 0))
        h = float(e.get("height", 0))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    pad = export_pad
    # Uniform scale: limit is whichever axis fills to edge first
    scale = min(
        png_width / (max_x - min_x + 2 * pad),
        png_height / (max_y - min_y + 2 * pad),
    )
    # Centre the scaled content in the PNG
    fitted_w = (max_x - min_x + 2 * pad) * scale
    fitted_h = (max_y - min_y + 2 * pad) * scale
    margin_x = (png_width - fitted_w) / 2
    margin_y = (png_height - fitted_h) / 2

    return {
        "scale": scale,
        "offset_x": margin_x + pad * scale - min_x * scale,
        "offset_y": margin_y + pad * scale - min_y * scale,
    }


def to_png_xy(ex: float, ey: float, t: dict) -> tuple[float, float]:
    return ex * t["scale"] + t["offset_x"], ey * t["scale"] + t["offset_y"]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return math.hypot(ax - bx, ay - by)


def padded(
    box: tuple[float, float, float, float],
    pad_x: float,
    pad_y: float,
    min_width: float = 30,
    min_height: float = 14,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width  = max(min_width,  (x2 - x1) + 2 * pad_x)
    height = max(min_height, (y2 - y1) + 2 * pad_y)
    return (
        round(cx - width  / 2),
        round(cy - height / 2),
        round(cx + width  / 2),
        round(cy + height / 2),
    )


# ---------------------------------------------------------------------------
# Build candidate list in PNG-pixel space
# ---------------------------------------------------------------------------

def collect_candidates(excal_data: dict, transform: dict) -> tuple[list[dict], list[dict]]:
    """Return (line_candidates, token_candidates) with bboxes in PNG pixels."""
    line_candidates: list[dict] = []
    token_candidates: list[dict] = []

    for element in excal_data.get("elements", []):
        if element.get("type") != "text" or element.get("isDeleted"):
            continue

        raw = element.get("originalText") or element.get("text") or ""
        if not raw.strip():
            continue

        ex = float(element.get("x", 0))
        ey = float(element.get("y", 0))
        ew = float(element.get("width", 0))
        eh = float(element.get("height", 0))
        stroke_color = element.get("strokeColor", "")
        is_black = (stroke_color == "#1e1e1e")

        # Large lineHeight means a tiny glyph inside a tall slot.
        # We use only the visual text height (≈ fontSize × 1.2) for the bbox
        # so that hotspots are not artificially bloated.
        element_line_height = float(element.get("lineHeight", 1.25) or 1.25)

        lines = raw.split("\n")
        total_lines = max(1, len(lines))
        line_h_excal = eh / total_lines  # height of one line slot in canvas units

        # Clip to visual text height (text renders at top of each slot)
        visual_h_excal = line_h_excal * min(1.0, 1.2 / element_line_height)

        for line_index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # per-line bbox using the VISUAL height, anchored at slot top
            ly1_excal = ey + line_index * line_h_excal

            lx1, ly1 = to_png_xy(ex,      ly1_excal, transform)
            lx2, ly2 = to_png_xy(ex + ew, ly1_excal + visual_h_excal, transform)

            line_candidates.append({"norm": norm(stripped), "bbox": (lx1, ly1, lx2, ly2)})

            # Skip black elements for TOKEN candidates — they are section headers
            # whose words (e.g. "Crossmatch", "Join") would shadow the real
            # function-list entries in coloured text.
            if is_black:
                continue

            line_text = line.rstrip()
            line_len  = max(1, len(line_text))

            for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", line_text):
                token_norm = norm(match.group(0))
                if not token_norm:
                    continue

                # proportional x within the element
                tx1_excal = ex + ew * (match.start() / line_len)
                tx2_excal = ex + ew * (match.end()   / line_len)

                tx1, ty1 = to_png_xy(tx1_excal, ly1_excal, transform)
                tx2, ty2 = to_png_xy(tx2_excal, ly1_excal + visual_h_excal, transform)

                token_candidates.append({"norm": token_norm, "bbox": (tx1, ty1, tx2, ty2)})

    return line_candidates, token_candidates


# ---------------------------------------------------------------------------
# Match each <area alt> to its best Excalidraw text candidate
# ---------------------------------------------------------------------------

ALIASES: dict[str, str] = {
    "getpartitionexecution": "getpartition",
}


def choose_bbox(
    alt_text: str,
    old_box: tuple[int, int, int, int],
    line_candidates: list[dict],
    token_candidates: list[dict],
) -> tuple[int, int, int, int]:
    alt_norm = ALIASES.get(norm(alt_text), norm(alt_text))

    # 1. Exact token match (single identifier)
    exact = [c for c in token_candidates if c["norm"] == alt_norm]
    if exact:
        best = min(exact, key=lambda c: distance(c["bbox"], old_box))
        return padded(best["bbox"], 8, 4)

    # 1b. Exact line match (needed for class-name labels like MapCatalog that
    # may only exist in black text lines and thus are absent from token matches)
    exact_line = [c for c in line_candidates if c["norm"] == alt_norm]
    if exact_line:
        best = min(exact_line, key=lambda c: distance(c["bbox"], old_box))
        return padded(best["bbox"], 8, 4)

    # 2. Partial token match — require the shorter side to cover ≥60 % of
    #    the longer side so that "catalog" doesn't match "margincatalog".
    min_overlap = 0.60
    partial = [
        c for c in token_candidates
        if (alt_norm in c["norm"] or c["norm"] in alt_norm)
        and len(c["norm"]) >= len(alt_norm) * min_overlap
        and len(alt_norm) >= len(c["norm"]) * min_overlap
    ]
    if partial:
        best = min(partial, key=lambda c: distance(c["bbox"], old_box))
        return padded(best["bbox"], 10, 4)

    # 3. Line-level match (multi-word labels like "head, tail"), same overlap rule
    line_m = [
        c for c in line_candidates
        if (alt_norm in c["norm"] or c["norm"] in alt_norm)
        and len(c["norm"]) >= len(alt_norm) * min_overlap
        and len(alt_norm) >= len(c["norm"]) * min_overlap
    ]
    if line_m:
        best = min(line_m, key=lambda c: distance(c["bbox"], old_box))
        return padded(best["bbox"], 14, 5, min_width=40, min_height=18)

    print(f"  WARNING: no match for '{alt_text}'")
    return old_box


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_update(config: UpdateConfig) -> UpdateResult:
    """Canonical function for regenerating API-surface hotspot coords."""
    excal_data = json.loads(config.excalidraw_path.read_text())
    index_text = config.index_path.read_text()

    elements = excal_data.get("elements", [])
    transform = compute_transform(
        elements,
        png_width=config.png_width,
        png_height=config.png_height,
        export_pad=config.export_pad,
    )

    line_cands, token_cands = collect_candidates(excal_data, transform)

    area_pattern = re.compile(
        r'(<area\s+shape="rect"\s+coords=")([0-9]+,[0-9]+,[0-9]+,[0-9]+)'
        r'("\s+href="[^"]+"\s+alt="([^"]+)"\s*/>)'
    )

    updated = 0
    unmatched = []

    def replace_area(match: re.Match) -> str:
        nonlocal updated
        old_box = tuple(map(int, match.group(2).split(",")))
        alt_text = match.group(4)
        new_box = choose_bbox(alt_text, old_box, line_cands, token_cands)
        if new_box != old_box:
            updated += 1
        else:
            unmatched.append(alt_text)
        return (
            f"{match.group(1)}{new_box[0]},{new_box[1]},{new_box[2]},{new_box[3]}"
            f"{match.group(3)}"
        )

    new_text = area_pattern.sub(replace_area, index_text)

    # Authored coords are now in PNG-pixel space == native space.
    # Set data-original-map-* equal to data-map-* so JS does no extra scaling.
    new_text = re.sub(
        r'data-original-map-width="[^"]*"',
        f'data-original-map-width="{config.png_width}"',
        new_text,
    )
    new_text = re.sub(
        r'data-original-map-height="[^"]*"',
        f'data-original-map-height="{config.png_height}"',
        new_text,
    )

    # Reset any leftover shrink-scale overrides.
    for old, new in [
        ('data-hitbox-scale-x="0.88"', 'data-hitbox-scale-x="1.0"'),
        ('data-hitbox-scale-y="0.42"', 'data-hitbox-scale-y="1.0"'),
        ('data-hitbox-scale-x="0.72"', 'data-hitbox-scale-x="1.0"'),
        ('data-hitbox-scale-y="0.62"', 'data-hitbox-scale-y="1.0"'),
    ]:
        new_text = new_text.replace(old, new)

    config.index_path.write_text(new_text)
    return UpdateResult(
        updated_areas=updated,
        unmatched_alts=tuple(unmatched),
        transform_scale=float(transform["scale"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate API surface map coords from Excalidraw")
    parser.add_argument("--excalidraw", type=pathlib.Path, default=DEFAULT_EXCALIDRAW_PATH)
    parser.add_argument("--index", type=pathlib.Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--png-width", type=int, default=3719)
    parser.add_argument("--png-height", type=int, default=2164)
    parser.add_argument("--export-pad", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = UpdateConfig(
        excalidraw_path=args.excalidraw,
        index_path=args.index,
        png_width=args.png_width,
        png_height=args.png_height,
        export_pad=args.export_pad,
    )

    result = run_update(config)
    print(f"updated areas: {result.updated_areas}")
    print(f"transform scale: {result.transform_scale:.4f}")
    if result.unmatched_alts:
        print(f"unmatched (kept old coords): {list(result.unmatched_alts)}")


if __name__ == "__main__":
    main()
