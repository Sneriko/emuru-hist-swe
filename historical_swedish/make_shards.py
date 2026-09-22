from __future__ import annotations

import argparse
import json
from pathlib import Path

import webdataset as wds
from PIL import Image
from tqdm import tqdm

from .common import image_bytes, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Emuru-compatible WebDataset shards.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for split-XXXXXX.tar shards")
    parser.add_argument("--mode", choices=["vae", "emuru"], required=True)
    parser.add_argument("--maxcount", type=int, default=8000)
    parser.add_argument("--image-field", default=None)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row.get("split", "train") == split]
        if not selected:
            continue
        pattern = str(args.output / f"{split}-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
            for row in tqdm(selected, desc=f"shard {split}"):
                field = args.image_field or ("pair_path" if args.mode == "emuru" else "normalized_path")
                image = Image.open(row[field])
                rgb = image_bytes(image, "RGB")
                bw = image_bytes(image, "L")
                metadata = dict(row)
                metadata["writer_id"] = int(metadata.get("writer_index", 0))
                sink.write({
                    "__key__": str(row["key"]),
                    "rgb.png": rgb,
                    "bw.png": bw,
                    "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
                })
        counts[split] = len(selected)
        shard_paths = sorted(args.output.glob(f"{split}-*.tar"))
        (args.output / f"{split}_shards.txt").write_text(
            "\n".join(str(path.resolve()) for path in shard_paths) + "\n", encoding="utf-8"
        )
    (args.output / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"counts={counts}")


if __name__ == "__main__":
    main()
