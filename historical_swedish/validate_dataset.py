from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifests before sharding/training.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-field", default="pair_path")
    parser.add_argument("--max-width", type=int, default=768)
    parser.add_argument("--expected-height", type=int, default=64)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    errors = Counter()
    keys = set()
    groups_by_split: dict[str, set[str]] = {}
    for row in rows:
        key = str(row.get("key", ""))
        if not key or key in keys:
            errors["missing_or_duplicate_key"] += 1
        keys.add(key)
        text = row.get("text", "")
        if not text or text != unicodedata.normalize("NFC", text):
            errors["empty_or_non_nfc_text"] += 1
        path = Path(row.get(args.image_field, ""))
        if not path.is_file():
            errors["missing_image"] += 1
            continue
        with Image.open(path) as image:
            if image.height != args.expected_height:
                errors["wrong_height"] += 1
            if image.width > args.max_width:
                errors["too_wide"] += 1
        split = row.get("split", "train")
        group = str(row.get("document_id", ""))
        groups_by_split.setdefault(split, set()).add(group)
    splits = sorted(groups_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                errors[f"document_leakage_{left}_{right}"] += len(overlap)
    print(f"rows={len(rows):,} errors={dict(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

