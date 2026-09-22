from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_jsonl, stable_fraction, write_jsonl


def century(year: str | int | None) -> str:
    try:
        value = int(str(year)[:4])
        return f"{((value - 1) // 100 + 1):02d}c"
    except (TypeError, ValueError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Make leakage-resistant group splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", choices=["document_id", "volume_id", "pseudo_writer_id"], default="document_id")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val-fraction + test-fraction must be < 1")
    rows = read_jsonl(args.input)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[args.group_by])].append(row)
    counts = Counter()
    output = []
    for group, members in groups.items():
        # Include dominant century in the hash key so each era receives its own deterministic allocation.
        era = Counter(century(row.get("year")) for row in members).most_common(1)[0][0]
        value = stable_fraction(f"{era}:{group}", args.seed)
        split = "test" if value < args.test_fraction else "val" if value < args.test_fraction + args.val_fraction else "train"
        for row in members:
            row["century"] = century(row.get("year"))
            row["split"] = split
            output.append(row)
            counts[(split, row["century"])] += 1
    write_jsonl(args.output, output)
    print(f"wrote {len(output):,} rows; split/century={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

