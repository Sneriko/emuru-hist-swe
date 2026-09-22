from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .common import read_jsonl, stable_fraction, write_jsonl


def paste_pair(style: Image.Image, target: Image.Image, gap: int, max_width: int) -> tuple[Image.Image, int]:
    style_limit = min(style.width, max(32, max_width // 2))
    style = style.crop((0, 0, style_limit, style.height))
    remaining = max_width - style.width - gap
    target = target.crop((0, 0, max(1, remaining), target.height))
    mode = "RGB" if style.mode == "RGB" or target.mode == "RGB" else "L"
    background = (255, 255, 255) if mode == "RGB" else 255
    canvas = Image.new(mode, (style.width + gap + target.width, max(style.height, target.height)), background)
    canvas.paste(style.convert(mode), (0, 0))
    canvas.paste(target.convert(mode), (style.width + gap, 0))
    return canvas, style.width + gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Create same-style prefix/continuation training pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--pair-by", choices=["page_id", "pseudo_writer_id"], default="page_id")
    parser.add_argument("--pairs-per-target", type=int, default=1)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--use", choices=["raw", "normalized"], default="normalized")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], str(row[args.pair_by]))].append(row)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    output = []
    skipped = Counter()
    for (split, group), members in groups.items():
        if len(members) < 2:
            skipped["singleton_group"] += len(members)
            continue
        for target in members:
            candidates = [row for row in members if row["key"] != target["key"]]
            rng = random.Random(args.seed + int(stable_fraction(target["key"], args.seed) * 2**31))
            for pair_index in range(args.pairs_per_target):
                style = candidates[pair_index % len(candidates)] if pair_index < len(candidates) else rng.choice(candidates)
                style_img = Image.open(style[f"{args.use}_path"])
                target_img = Image.open(target[f"{args.use}_path"])
                paired, prefix_width = paste_pair(style_img, target_img, args.gap, args.max_width)
                key = f"{target['key']}-{pair_index:02d}"
                pair_path = args.image_dir / f"{key}.png"
                paired.save(pair_path)
                output.append({
                    **target,
                    "key": key,
                    "pair_path": str(pair_path.resolve()),
                    "style_key": style["key"],
                    "style_text": style["text"],
                    "gen_text": target["text"],
                    "text": f"{style['text']} {target['text']}",
                    "style_prefix_width": prefix_width,
                    "pair_group": group,
                    "pair_by": args.pair_by,
                    "split": split,
                })
    write_jsonl(args.output, output)
    print(f"wrote {len(output):,} pairs; skipped={dict(skipped)}")


if __name__ == "__main__":
    main()

