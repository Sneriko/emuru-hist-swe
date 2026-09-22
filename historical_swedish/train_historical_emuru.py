from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from tqdm import tqdm

from custom_datasets import DataLoaderManager
from models.emuru import Emuru


def resolve_shards(pattern: str) -> str | list[str]:
    path = Path(pattern)
    if path.is_file() and path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pattern


def forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, loader, max_batches: int) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        loss, _, _ = model(batch["img"], input_ids=batch["input_ids"].long(), attention_mask=batch["attention_mask"])
        total += loss.item()
        seen += 1
        if seen >= max_batches:
            break
    model.train()
    return total / max(1, seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained Emuru with historical data and optional FontSquare replay.")
    parser.add_argument("--historical-train-pattern", required=True)
    parser.add_argument("--historical-val-pattern", required=True)
    parser.add_argument("--replay-train-pattern")
    parser.add_argument("--base-model", default="blowing-up-groundhogs/emuru")
    parser.add_argument("--vae-path", help="Optional adapted VAE; replaces the VAE bundled/referenced by the base model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--historical-probability", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--teacher-noise", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.historical_probability <= 1:
        raise ValueError("historical-probability must be in [0, 1]")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    model = Emuru.from_pretrained(args.base_model)
    if args.vae_path:
        model.vae = AutoencoderKL.from_pretrained(args.vae_path)
        model.config.vae_name_or_path = args.vae_path
        model.set_training(model.vae, False)

    historical = DataLoaderManager(
        resolve_shards(args.historical_train_pattern),
        resolve_shards(args.historical_val_pattern),
        args.batch_size,
        args.eval_batch_size,
        args.workers,
        False,
        False,
        tokenizer=model.tokenizer,
    )
    train_historical = historical.create_dataset("train", "t5")
    val_historical = historical.create_dataset("eval", "t5")
    train_replay = None
    if args.replay_train_pattern:
        replay = DataLoaderManager(
            resolve_shards(args.replay_train_pattern),
            resolve_shards(args.historical_val_pattern),
            args.batch_size,
            args.eval_batch_size,
            args.workers,
            False,
            False,
            tokenizer=model.tokenizer,
        )
        train_replay = replay.create_dataset("train", "t5")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))
    if train_replay is None:
        model, optimizer, train_historical, val_historical, scheduler = accelerator.prepare(
            model, optimizer, train_historical, val_historical, scheduler
        )
    else:
        model, optimizer, train_historical, val_historical, train_replay, scheduler = accelerator.prepare(
            model, optimizer, train_historical, val_historical, train_replay, scheduler
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical_iter = forever(train_historical)
    replay_iter = forever(train_replay) if train_replay is not None else None
    rng = random.Random(args.seed + accelerator.process_index)
    progress = tqdm(total=args.epochs * args.steps_per_epoch, disable=not accelerator.is_local_main_process)
    for epoch in range(args.epochs):
        model.train()
        for _ in range(args.steps_per_epoch):
            use_historical = replay_iter is None or rng.random() < args.historical_probability
            batch = next(historical_iter if use_historical else replay_iter)
            with accelerator.accumulate(model):
                loss, _, _ = model(
                    batch["img"],
                    input_ids=batch["input_ids"].long(),
                    attention_mask=batch["attention_mask"],
                    noise=args.teacher_noise,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", source="historical" if use_historical else "replay")
        val_loss = evaluate(model, val_historical, args.max_eval_batches)
        if accelerator.is_main_process:
            print(f"epoch={epoch + 1} historical_val_loss={val_loss:.6f}")
            accelerator.unwrap_model(model).save_pretrained(args.output_dir / f"epoch-{epoch + 1:03d}")
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
