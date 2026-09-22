from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision import transforms
from tqdm import tqdm

from custom_datasets.load_hf_dataset import DataProcessor
from models.writer_id import WriterID

from .common import read_jsonl, write_jsonl


@torch.inference_mode()
def page_embedding(model, rows: list[dict], device: torch.device, lines_per_page: int) -> np.ndarray:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    chosen = sorted(rows, key=lambda row: row["key"])[:lines_per_page]
    images = [transform(Image.open(row["normalized_path"]).convert("L")) for row in chosen]
    batch = DataProcessor.pad_images_fixed(images).to(device)
    features = model.compute_features(batch).mean(dim=(-1, -2))
    embedding = features.mean(dim=0)
    embedding = torch.nn.functional.normalize(embedding, dim=0)
    return embedding.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign conservative pseudo-writer IDs within provenance units.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--writer-model", default="pretrained_models/emuru_vae_writer_id")
    parser.add_argument("--within", choices=["volume_id", "document_id"], default="volume_id")
    parser.add_argument("--distance-threshold", type=float, default=0.18)
    parser.add_argument("--lines-per-page", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    by_page: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_page[(str(row[args.within]), str(row["page_id"]))].append(row)

    device = torch.device(args.device)
    model = WriterID.from_pretrained(args.writer_model).to(device).eval()
    model.remove_last_layers()
    embeddings: dict[tuple[str, str], np.ndarray] = {}
    for key, page_rows in tqdm(by_page.items(), desc="writer embeddings"):
        embeddings[key] = page_embedding(model, page_rows, device, args.lines_per_page)

    labels: dict[tuple[str, str], str] = {}
    units = defaultdict(list)
    for unit, page_id in by_page:
        units[unit].append(page_id)
    for unit, pages in units.items():
        pages = sorted(pages)
        if len(pages) == 1:
            predicted = np.array([0])
        else:
            matrix = np.stack([embeddings[(unit, page)] for page in pages])
            predicted = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=args.distance_threshold,
            ).fit_predict(matrix)
        for page, label in zip(pages, predicted):
            labels[(unit, page)] = f"{unit}:pseudo-writer-{int(label):04d}"
    for row in rows:
        row["pseudo_writer_id"] = labels[(str(row[args.within]), str(row["page_id"]))]
        row["pseudo_writer_method"] = "emuru_writer_embedding_agglomerative"
        row["pseudo_writer_distance_threshold"] = args.distance_threshold
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows):,} lines with {len(set(labels.values())):,} pseudo-writer IDs")


if __name__ == "__main__":
    main()

