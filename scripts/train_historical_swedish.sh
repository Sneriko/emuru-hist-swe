#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/data/riksarkivet/emuru_historical}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/models}"
VAE_STEPS_PER_EPOCH="${VAE_STEPS_PER_EPOCH:-1000}"
EMURU_STEPS_PER_EPOCH="${EMURU_STEPS_PER_EPOCH:-5000}"
FONT_SQUARE_REPLAY="${FONT_SQUARE_REPLAY:-https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000000..000048}.tar}"

# Stage A: optional. Run this only if the base VAE reconstruction benchmark is
# materially worse on historical lines than on FontSquare/modern handwriting.
accelerate launch --config_file configs/historical_swedish/accelerate_2gpu.yaml \
  -m historical_swedish.train_historical_vae \
  --train-pattern "$WORK_ROOT/shards_vae/train_shards.txt" \
  --val-pattern "$WORK_ROOT/shards_vae/val_shards.txt" \
  --output-dir "$OUTPUT_ROOT/vae" \
  --steps-per-epoch "$VAE_STEPS_PER_EPOCH"

# Stage B: Swedish historical autoregressive adaptation with 20% generic replay.
accelerate launch --config_file configs/historical_swedish/accelerate_2gpu.yaml \
  -m historical_swedish.train_historical_emuru \
  --historical-train-pattern "$WORK_ROOT/shards_emuru/train_shards.txt" \
  --historical-val-pattern "$WORK_ROOT/shards_emuru/val_shards.txt" \
  --replay-train-pattern "$FONT_SQUARE_REPLAY" \
  --base-model blowing-up-groundhogs/emuru \
  --vae-path "$OUTPUT_ROOT/vae/final" \
  --output-dir "$OUTPUT_ROOT/emuru" \
  --steps-per-epoch "$EMURU_STEPS_PER_EPOCH" \
  --historical-probability 0.8
