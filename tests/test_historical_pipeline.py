from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from historical_swedish.common import crop_polygon, normalize_background, read_jsonl, write_jsonl
from historical_swedish.extract_page_lines import extract
from historical_swedish.sample_smoke_dataset import allocate_pages, evenly_spaced, sample


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


def test_crop_polygon_masks_background_white() -> None:
    image = Image.new("RGB", (20, 20), "black")

    crop = crop_polygon(image, [(5, 5), (15, 5), (10, 15)], padding=2)

    assert crop.size == (15, 15)
    assert crop.getpixel((0, 0)) == (255, 255, 255)
    assert crop.getpixel((7, 7)) == (0, 0, 0)
    assert crop.getpixel((14, 14)) == (255, 255, 255)
    normalized = normalize_background(crop)
    assert normalized.getpixel((0, 0)) == 255
    assert normalized.getpixel((14, 14)) == 255


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
    assert all(row["height"] == 64 for row in rows)
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


def test_smoke_sampler_balances_volumes_and_preserves_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for volume_name, count in (("volume_a", 3), ("volume_b", 7)):
        volume = source / volume_name
        (volume / "page").mkdir(parents=True)
        for number in range(1, count + 1):
            stem = f"page_{number:04d}"
            Image.new("RGB", (600, 160), "ivory").save(volume / f"{stem}.png")
            xml = PAGE_XML.replace("page_0001.png", f"{stem}.png")
            (volume / "page" / f"{stem}.xml").write_text(xml, encoding="utf-8")

    output = tmp_path / "smoke"
    selected = sample(argparse.Namespace(
        root=source,
        output=output,
        pages=5,
        xml_glob="*.xml",
        min_lines=2,
        overwrite=False,
    ))

    assert [sum(page.volume.name == name for page in selected) for name in ("volume_a", "volume_b")] == [1, 4]
    assert len(list(output.glob("*/page/*.xml"))) == 5
    assert len(list(output.glob("*/*.png"))) == 5
    assert len((output / "smoke_selection.csv").read_text(encoding="utf-8").splitlines()) == 6
    assert allocate_pages([3, 7], 5) == [1, 4]
    assert [page.xml_path.stem for page in evenly_spaced(selected, 2)] == ["page_0001", "page_0005"]


def test_smoke_sampler_supports_direct_script_invocation(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "historical_swedish" / "sample_smoke_dataset.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a deterministic, volume-balanced" in result.stdout
