import argparse
import re
import sys
from pathlib import Path

AREA_PATTERN = re.compile(
    r'<area\s+[^>]*coords="(?P<coords>\d+,\d+,\d+,\d+)"[^>]*href="(?P<href>[^"]+)"[^>]*alt="(?P<alt>[^"]+)"[^>]*/?>'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate API surface image-map areas in docs/index.rst")
    parser.add_argument("--index", default="docs/index.rst", help="Path to index.rst containing the map")
    parser.add_argument(
        "--base-width", type=int, default=3719, help="Coordinate-space width (default: 3719 = PNG native)"
    )
    parser.add_argument(
        "--base-height", type=int, default=2164, help="Coordinate-space height (default: 2164 = PNG native)"
    )
    return parser.parse_args()


def validate(index_path: Path, base_width: int, base_height: int) -> list[str]:
    text = index_path.read_text(encoding="utf-8")
    errors: list[str] = []

    areas = list(AREA_PATTERN.finditer(text))
    if not areas:
        errors.append("No <area> entries found in API surface map.")
        return errors

    for i, match in enumerate(areas, start=1):
        coords_text = match.group("coords")
        href = match.group("href")
        alt = match.group("alt")
        x1, y1, x2, y2 = map(int, coords_text.split(","))

        if not alt.strip():
            errors.append(f"Area #{i} has empty alt text.")
        if x1 >= x2 or y1 >= y2:
            errors.append(f"Area #{i} has invalid rectangle coords: {coords_text}")
        if x1 < 0 or y1 < 0:
            errors.append(f"Area #{i} has negative coordinates: {coords_text}")
        if x2 > base_width or y2 > base_height:
            errors.append(f"Area #{i} exceeds base bounds ({base_width}x{base_height}): {coords_text}")
        if not href.endswith(".html"):
            errors.append(f"Area #{i} href is not an html target: {href}")

    return errors


def main() -> int:
    args = parse_args()
    index_path = Path(args.index)

    if not index_path.exists():
        print(f"ERROR: file not found: {index_path}")
        return 2

    errors = validate(index_path, args.base_width, args.base_height)
    if errors:
        print("API surface map validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("API surface map validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
