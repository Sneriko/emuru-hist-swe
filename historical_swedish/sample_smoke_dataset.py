from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .common import find_page_image, normalize_text
from .extract_page_lines import first_descendant, local_name


@dataclass(frozen=True)
class Page:
    xml_path: Path
    image_path: Path
    volume: Path
    line_count: int


def natural_key(path: Path) -> list[tuple[int, object]]:
    """Sort page paths in human order (page_2 before page_10)."""
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def inspect_page(xml_path: Path, root: Path, min_lines: int) -> Page | None:
    """Return a usable PAGE/image pair, or None for an unsuitable page."""
    try:
        page = first_descendant(ET.parse(xml_path).getroot(), "Page")
        if page is None:
            return None
        image_path = find_page_image(xml_path, page.attrib.get("imageFilename"))
        image_path.relative_to(root)
        with Image.open(image_path) as image:
            image.verify()
        usable_lines = 0
        for line in (node for node in page.iter() if local_name(node.tag) == "TextLine"):
            coords = first_descendant(line, "Coords")
            text = first_descendant(line, "Unicode")
            if (
                coords is not None
                and coords.attrib.get("points")
                and text is not None
                and text.text is not None
                and len(normalize_text(text.text)) >= 2
            ):
                usable_lines += 1
        if usable_lines < min_lines:
            return None
    except (ET.ParseError, OSError, ValueError):
        return None

    volume = xml_path.parent.parent if xml_path.parent.name.casefold() == "page" else xml_path.parent
    return Page(xml_path, image_path, volume, usable_lines)


def allocate_pages(sizes: list[int], target: int) -> list[int]:
    """Proportionally allocate target pages while retaining every volume."""
    if target < len(sizes):
        raise ValueError(f"target {target} is smaller than the {len(sizes)} volumes")
    target = min(target, sum(sizes))
    allocation = [1 for _ in sizes]
    remaining = target - len(sizes)
    capacities = [size - 1 for size in sizes]
    while remaining:
        total_capacity = sum(capacities)
        raw = [remaining * capacity / total_capacity for capacity in capacities]
        additions = [min(capacity, math.floor(value)) for capacity, value in zip(capacities, raw)]
        if not any(additions):
            best = max(range(len(sizes)), key=lambda index: (raw[index], capacities[index], -index))
            additions[best] = 1
        used = min(remaining, sum(additions))
        if used < sum(additions):
            for index in sorted(range(len(sizes)), key=lambda i: (raw[i] % 1, capacities[i]), reverse=True):
                while additions[index] and sum(additions) > remaining:
                    additions[index] -= 1
        for index, addition in enumerate(additions):
            allocation[index] += addition
            capacities[index] -= addition
        remaining -= sum(additions)
    return allocation


def evenly_spaced(items: list[Page], count: int) -> list[Page]:
    """Choose midpoint quantiles, avoiding a bias toward volume front matter."""
    if count >= len(items):
        return items
    return [items[math.floor((index + 0.5) * len(items) / count)] for index in range(count)]


def sample(args: argparse.Namespace) -> list[Page]:
    root = args.root.resolve()
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must not be the source directory or one of its children")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; use --overwrite to replace it")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)

    candidates = sorted(root.rglob(args.xml_glob), key=natural_key)
    pages = [page for path in candidates if (page := inspect_page(path, root, args.min_lines))]
    if not pages:
        raise FileNotFoundError(f"no usable PAGE XML/image pairs found under {root}")

    by_volume: dict[Path, list[Page]] = {}
    for page in pages:
        by_volume.setdefault(page.volume, []).append(page)
    volumes = sorted(by_volume, key=natural_key)
    allocation = allocate_pages([len(by_volume[volume]) for volume in volumes], args.pages)
    selected = [
        page
        for volume, count in zip(volumes, allocation)
        for page in evenly_spaced(by_volume[volume], count)
    ]

    output.mkdir(parents=True, exist_ok=True)
    copied: set[Path] = set()
    for page in selected:
        for source in (page.xml_path, page.image_path):
            relative = source.relative_to(root)
            if relative in copied:
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.add(relative)

    with (output / "smoke_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("volume_id", "xml_path", "image_path", "usable_lines"))
        writer.writeheader()
        for page in selected:
            writer.writerow({
                "volume_id": page.volume.relative_to(root).as_posix() or "root",
                "xml_path": page.xml_path.relative_to(root).as_posix(),
                "image_path": page.image_path.relative_to(root).as_posix(),
                "usable_lines": page.line_count,
            })
    print(
        f"copied {len(selected):,} pages from {len(volumes):,} volumes to {output} "
        f"({len(candidates) - len(pages):,} unsuitable pages skipped)"
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, volume-balanced PAGE-XML smoke-test dataset."
    )
    parser.add_argument("--root", type=Path, required=True, help="Source dataset tree")
    parser.add_argument("--output", type=Path, required=True, help="Target dataset tree")
    parser.add_argument("--pages", type=int, default=2000, help="Desired page count (default: 2000)")
    parser.add_argument("--xml-glob", default="*.xml", help="PAGE filename glob (default: *.xml)")
    parser.add_argument("--min-lines", type=int, default=2, help="Minimum usable text lines per page")
    parser.add_argument("--overwrite", action="store_true", help="Allow copying into a non-empty target")
    args = parser.parse_args()
    if args.pages < 1 or args.min_lines < 1:
        parser.error("--pages and --min-lines must be positive")
    sample(args)


if __name__ == "__main__":
    main()
