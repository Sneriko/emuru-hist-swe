#!/usr/bin/env bash
set -euo pipefail

# Override these without editing the script, for example:
# DATA_ROOT=/data/ra/pagexml WORK_ROOT=/data/ra/emuru bash scripts/prepare_historical_swedish.sh
DATA_ROOT="${DATA_ROOT:-/data/riksarkivet/pagexml_root}"
WORK_ROOT="${WORK_ROOT:-/data/riksarkivet/emuru_historical}"
METADATA_CSV="${METADATA_CSV:-}"
PAIR_BY="${PAIR_BY:-page_id}"

metadata_args=()
if [[ -n "$METADATA_CSV" ]]; then
  metadata_args=(--metadata-csv "$METADATA_CSV")
fi

python -m historical_swedish.extract_page_lines \
  --root "$DATA_ROOT" \
  --output "$WORK_ROOT/extracted" \
  "${metadata_args[@]}"

python -m historical_swedish.build_splits \
  --input "$WORK_ROOT/extracted/lines.jsonl" \
  --output "$WORK_ROOT/lines_split.jsonl" \
  --group-by document_id

python -m historical_swedish.build_pairs \
  --input "$WORK_ROOT/lines_split.jsonl" \
  --output "$WORK_ROOT/pairs.jsonl" \
  --image-dir "$WORK_ROOT/pairs" \
  --pair-by "$PAIR_BY" \
  --use normalized

python -m historical_swedish.validate_dataset \
  --input "$WORK_ROOT/pairs.jsonl" \
  --image-field pair_path

python -m historical_swedish.make_shards \
  --input "$WORK_ROOT/lines_split.jsonl" \
  --output "$WORK_ROOT/shards_vae" \
  --mode vae

python -m historical_swedish.make_shards \
  --input "$WORK_ROOT/pairs.jsonl" \
  --output "$WORK_ROOT/shards_emuru" \
  --mode emuru

