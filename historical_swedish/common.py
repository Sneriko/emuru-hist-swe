from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2")


def stable_fraction(value: str, seed: int = 24) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def normalize_text(text: str, form: str = "NFC") -> str:
    text = unicodedata.normalize(form, text)
    return re.sub(r"\s+", " ", text).strip()


def parse_points(points: str) -> list[tuple[int, int]]:
    result = []
    for pair in points.split():
        x, y = pair.split(",")
        result.append((round(float(x)), round(float(y))))
    return result


def find_page_image(xml_path: Path, image_filename: str | None) -> Path:
    candidates: list[Path] = []
    if image_filename:
        image_path = Path(image_filename)
        candidates.extend([xml_path.parent / image_path, xml_path.parent.parent / image_path])
    stem = xml_path.stem
    if xml_path.parent.name.lower() == "page":
        parents = [xml_path.parent.parent, xml_path.parent]
    else:
        parents = [xml_path.parent]
    for parent in parents:
        candidates.extend(parent / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No page image found for {xml_path} (PAGE imageFilename={image_filename!r})")


def crop_polygon(image: Image.Image, points: list[tuple[int, int]], padding: int = 8) -> Image.Image:
    xs, ys = zip(*points)
    left = max(0, min(xs) - padding)
    top = max(0, min(ys) - padding)
    right = min(image.width, max(xs) + padding + 1)
    bottom = min(image.height, max(ys) + padding + 1)
    return image.crop((left, top, right, bottom))


def to_line_height(image: Image.Image, height: int = 64, max_width: int = 768) -> Image.Image:
    image = image.convert("RGB")
    width = max(1, round(image.width * height / max(1, image.height)))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if width > max_width:
        image = image.crop((0, 0, max_width, height))
    return image


def normalize_background(image: Image.Image) -> Image.Image:
    """Conservative grayscale illumination correction; deliberately no binarization."""
    gray = ImageOps.grayscale(image)
    background = gray.filter(ImageFilter.GaussianBlur(radius=max(3, gray.height // 8)))
    arr = np.asarray(gray, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    corrected = np.clip(arr - bg + 235.0, 0, 255).astype(np.uint8)
    result = Image.fromarray(corrected, mode="L")
    result = ImageOps.autocontrast(result, cutoff=(0.2, 0.2))
    return ImageEnhance.Contrast(result).enhance(1.05)


def image_bytes(image: Image.Image, mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    image.convert(mode).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

