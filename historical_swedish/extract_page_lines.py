from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .common import (
    LINE_HEIGHT,
    MAX_LINE_WIDTH,
    crop_polygon,
    find_page_image,
    normalize_background,
    normalize_text,
    parse_points,
    to_line_height,
    write_jsonl,
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in element.iter() if local_name(node.tag) == name), None)


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = row.get("xml_path") or row.get("page_id")
        if not key:
            raise ValueError("metadata CSV needs xml_path or page_id")
        result[Path(key).as_posix()] = {k: v for k, v in row.items() if v not in (None, "")}
        result[Path(key).name] = result[Path(key).as_posix()]
    return result


def page_metadata(xml_path: Path, root: Path, supplied: dict[str, dict[str, str]]) -> dict[str, str]:
    relative = xml_path.relative_to(root).as_posix()
    meta = supplied.get(relative, supplied.get(xml_path.name, {})).copy()
    parts = xml_path.relative_to(root).parts
    fallback_document = "/".join(parts[:-2] if xml_path.parent.name.lower() == "page" else parts[:-1]) or "root"
    page_id = meta.get("page_id", relative)
    meta.setdefault("archive_id", parts[0] if len(parts) > 1 else "unknown")
    meta.setdefault("document_id", fallback_document)
    meta.setdefault("volume_id", meta["document_id"])
    meta.setdefault("page_id", page_id)
    meta.setdefault("pseudo_writer_id", meta.get("writer_id", page_id))
    meta.setdefault("year", "")
    return meta


def extract(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    out = args.output.resolve()
    raw_dir = out / "lines" / "raw"
    norm_dir = out / "lines" / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    supplied = load_metadata(args.metadata_csv)
    xml_files = sorted(root.rglob(args.xml_glob))
    if not xml_files:
        raise FileNotFoundError(f"No PAGE XML matching {args.xml_glob!r} under {root}")

    rows = []
    charset: Counter[str] = Counter()
    skipped = Counter()
    for xml_path in tqdm(xml_files, desc="PAGE XML"):
        try:
            tree = ET.parse(xml_path)
            page = first_descendant(tree.getroot(), "Page")
            if page is None:
                raise ValueError("missing Page element")
            image_path = find_page_image(xml_path, page.attrib.get("imageFilename"))
            image = Image.open(image_path).convert("RGB")
            meta = page_metadata(xml_path, root, supplied)
        except Exception as exc:
            if args.strict:
                raise
            skipped[type(exc).__name__] += 1
            continue

        line_index = 0
        for line in (node for node in page.iter() if local_name(node.tag) == "TextLine"):
            coords = first_descendant(line, "Coords")
            unicode_node = first_descendant(line, "Unicode")
            if coords is None or not coords.attrib.get("points") or unicode_node is None or unicode_node.text is None:
                skipped["incomplete_line"] += 1
                continue
            text = normalize_text(unicode_node.text, args.unicode_form)
            if len(text) < args.min_chars:
                skipped["short_text"] += 1
                continue
            try:
                points = parse_points(coords.attrib["points"])
                raw = to_line_height(crop_polygon(image, points, args.padding), args.height, args.max_width)
            except Exception:
                if args.strict:
                    raise
                skipped["bad_polygon"] += 1
                continue
            if raw.width < args.min_width:
                skipped["narrow_crop"] += 1
                continue

            key = f"{len(rows):09d}"
            raw_path = raw_dir / f"{key}.png"
            norm_path = norm_dir / f"{key}.png"
            raw.save(raw_path)
            normalize_background(raw).save(norm_path)
            charset.update(text)
            rows.append({
                **meta,
                "key": key,
                "line_id": line.attrib.get("id", f"line-{line_index}"),
                "xml_path": str(xml_path),
                "image_path": str(image_path),
                "raw_path": str(raw_path),
                "normalized_path": str(norm_path),
                "text": text,
                "polygon": points,
                "width": raw.width,
                "height": raw.height,
            })
            line_index += 1

    write_jsonl(out / "lines.jsonl", rows)
    (out / "charset.txt").write_text("".join(sorted(charset)), encoding="utf-8")
    with (out / "charset_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("codepoint\tcharacter\tcount\n")
        for char, count in sorted(charset.items(), key=lambda item: ord(item[0])):
            handle.write(f"U+{ord(char):04X}\t{char.replace(chr(9), '<TAB>')}\t{count}\n")
    print(f"wrote {len(rows):,} lines to {out / 'lines.jsonl'}; skipped={dict(skipped)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract masked line crops and text from PAGE-XML.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--xml-glob", default="*.xml")
    parser.add_argument("--height", type=int, default=LINE_HEIGHT)
    parser.add_argument("--max-width", type=int, default=MAX_LINE_WIDTH)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--unicode-form", choices=["NFC", "NFKC"], default="NFC")
    parser.add_argument("--strict", action="store_true")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
