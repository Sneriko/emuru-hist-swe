from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from historical_swedish.common import read_jsonl, write_jsonl
from historical_swedish.extract_page_lines import extract


PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page_0001.png" imageWidth="600" imageHeight="160">
    <TextRegion id="r1"><Coords points="0,0 599,0 599,159 0,159"/>
      <TextLine id="l1"><Coords points="10,10 500,10 500,65 10,65"/><TextEquiv><Unicode>År 1752 föddes ett barn</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Coords points="20,85 560,85 560,145 20,145"/><TextEquiv><Unicode>Öfver ån gick prästen</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def test_page_to_pairs_and_shards(tmp_path: Path) -> None:
    root = tmp_path / "source" / "volume_a"
    page_dir = root / "page"
    page_dir.mkdir(parents=True)
    image = Image.new("RGB", (600, 160), "ivory")
    draw = ImageDraw.Draw(image)
    draw.line((10, 40, 500, 40), fill="black", width=3)
    draw.line((20, 115, 560, 115), fill="black", width=3)
    image.save(root / "page_0001.png")
    (page_dir / "page_0001.xml").write_text(PAGE_XML, encoding="utf-8")

    extracted = tmp_path / "extracted"
    extract(argparse.Namespace(
        root=tmp_path / "source",
        output=extracted,
        metadata_csv=None,
        xml_glob="*.xml",
        height=64,
        max_width=768,
        padding=8,
        min_width=24,
        min_chars=2,
        unicode_form="NFC",
        strict=True,
    ))
    rows = read_jsonl(extracted / "lines.jsonl")
    assert len(rows) == 2
    assert "Å" in (extracted / "charset.txt").read_text(encoding="utf-8")
    for row in rows:
        row["split"] = "train"
    split_manifest = tmp_path / "split.jsonl"
    write_jsonl(split_manifest, rows)

    pairs = tmp_path / "pairs.jsonl"
    subprocess.run([
        sys.executable, "-m", "historical_swedish.build_pairs",
        "--input", str(split_manifest),
        "--output", str(pairs),
        "--image-dir", str(tmp_path / "pairs"),
    ], check=True)
    pair_rows = read_jsonl(pairs)
    assert len(pair_rows) == 2
    assert all(row["style_key"] != row["key"].split("-")[0] for row in pair_rows)

    shard_dir = tmp_path / "shards"
    subprocess.run([
        sys.executable, "-m", "historical_swedish.make_shards",
        "--input", str(pairs),
        "--output", str(shard_dir),
        "--mode", "emuru",
        "--maxcount", "1",
    ], check=True)
    assert len(list(shard_dir.glob("train-*.tar"))) == 2
    listed = (shard_dir / "train_shards.txt").read_text(encoding="utf-8").splitlines()
    assert len(listed) == 2
    assert json.loads((shard_dir / "counts.json").read_text(encoding="utf-8"))["train"] == 2

