from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import webdataset as wds
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from custom_datasets.load_hf_dataset import DataProcessor


def resolve_shards(pattern: str) -> str | list[str]:
    path = Path(pattern)
    if path.is_file() and path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pattern


def make_loader(pattern: str, batch_size: int, workers: int, train: bool) -> DataLoader:
    transform_rgb = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    transform_bw = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = wds.WebDataset(
        resolve_shards(pattern), nodesplitter=wds.split_by_node, shardshuffle=100 if train else False
    )
    if train:
        dataset = dataset.shuffle(1000)
    dataset = (
        dataset.decode("pil").map(lambda sample: {
            "rgb": transform_rgb(sample["rgb.png"].convert("RGB")),
            "bw": transform_bw(sample["bw.png"].convert("L")),
        })
    )

    def collate(batch):
        return {
            "rgb": DataProcessor.pad_images_fixed([row["rgb"] for row in batch]),
            "bw": DataProcessor.pad_images_fixed([row["bw"] for row in batch]),
        }

    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate, drop_last=train)


@torch.no_grad()
def evaluate(vae, loader, accelerator, max_batches: int) -> dict[str, float]:
    vae.eval()
    totals = {"l1": 0.0, "kl": 0.0}
    seen = 0
    for batch in loader:
        posterior = vae.encode(batch["rgb"]).latent_dist
        reconstruction = vae.decode(posterior.mode()).sample
        totals["l1"] += F.l1_loss(reconstruction, batch["bw"]).item()
        totals["kl"] += posterior.kl().mean().item()
        seen += 1
        if seen >= max_batches:
            break
    vae.train()
    return {key: value / max(1, seen) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-rate domain adaptation of Emuru's image VAE.")
    parser.add_argument("--train-pattern", required=True)
    parser.add_argument("--val-pattern", required=True)
    parser.add_argument("--base-vae", default="blowing-up-groundhogs/emuru_vae")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    vae = AutoencoderKL.from_pretrained(args.base_vae)
    vae.requires_grad_(True)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))
    train_loader = make_loader(args.train_pattern, args.batch_size, args.workers, True)
    val_loader = make_loader(args.val_pattern, args.batch_size, 0, False)
    vae, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        vae, optimizer, train_loader, val_loader, scheduler
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=args.epochs * args.steps_per_epoch, disable=not accelerator.is_local_main_process)
    for epoch in range(args.epochs):
        vae.train()
        for step, batch in enumerate(train_loader):
            if step >= args.steps_per_epoch:
                break
            with accelerator.accumulate(vae):
                posterior = vae.encode(batch["rgb"]).latent_dist
                latent = posterior.sample()
                reconstruction = vae.decode(latent).sample
                l1 = F.l1_loss(reconstruction, batch["bw"])
                dx_pred = reconstruction[..., 1:] - reconstruction[..., :-1]
                dx_true = batch["bw"][..., 1:] - batch["bw"][..., :-1]
                dy_pred = reconstruction[..., 1:, :] - reconstruction[..., :-1, :]
                dy_true = batch["bw"][..., 1:, :] - batch["bw"][..., :-1, :]
                edge = F.l1_loss(dx_pred, dx_true) + F.l1_loss(dy_pred, dy_true)
                kl = posterior.kl().mean()
                loss = l1 + args.edge_weight * edge + args.kl_weight * kl
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1.item():.4f}")
        metrics = evaluate(vae, val_loader, accelerator, args.max_eval_batches)
        if accelerator.is_main_process:
            print(f"epoch={epoch + 1} val={metrics}")
            accelerator.unwrap_model(vae).save_pretrained(args.output_dir / f"epoch-{epoch + 1:03d}")
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(vae).save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
