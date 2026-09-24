# Historical Swedish Emuru extension

This directory is a runnable research scaffold built on commit
`d8db91a606297ae0b173ec90de03e31fd927ba91` of the official
`aimagelab/Emuru-autoregressive-text-img` repository. It preserves the upstream
code and adds PAGE-XML preparation, leakage-resistant splits, prefix/continuation
pair construction, WebDataset export, VAE domain adaptation, and pretrained
Emuru fine-tuning with FontSquare replay.

## What is—and is not—learned by the VAE

Do not retrain the VAE merely because the corpus contains Swedish words or
`å`, `ä`, and `ö`. Emuru conditions on `google/byt5-small`, whose byte-level
tokenizer can represent those characters already. The VAE sees pixels and
learns image statistics: stroke shapes, ink, background, degradation, and line
geometry. Adapt the VAE only if reconstruction tests show that its latent space
loses historically important strokes or creates poor reconstructions.

The upstream VAE's auxiliary HTR model *does* use a fixed character alphabet.
If you reproduce the full upstream VAE objective with an HTR loss, train a new
HTR auxiliary model whose alphabet is generated from `charset.txt`; otherwise
the HTR loss cannot accept corpus-specific characters. The supplied historical
VAE trainer deliberately uses reconstruction + edge + KL losses and therefore
does not pretend that the released English-biased HTR head supports every
historical transcription symbol.

Recommended ablation:

1. released VAE + historical Emuru fine-tuning;
2. adapted VAE + historical Emuru fine-tuning;
3. optionally, adapted VAE with a separately trained Swedish-aware HTR loss.

Keep (1) as the baseline. VAE fine-tuning can improve historical reconstruction,
but it can also shift the latent distribution that the released Emuru model was
trained to predict.

## Expected source layout

The extractor recursively locates PAGE-XML. It supports the common layout:

```text
dataset/
  collection_or_volume/
    page_0001.jpg
    page/
      page_0001.xml
```

It also reads PAGE `imageFilename`. A metadata CSV is strongly recommended;
copy `configs/historical_swedish/metadata.example.csv` and add one row per PAGE
file. The minimum useful metadata are `document_id`, `volume_id`, `page_id`, and
`year`. If known, add `writer_id` or `pseudo_writer_id`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-historical.txt
```

The upstream pins CUDA-enabled PyTorch packages. On an offline/on-prem system,
install matching wheels from your internal mirror before the remaining
requirements.

## Data preparation

Run the complete default path:

```bash
DATA_ROOT=/data/ra/pagexml \
WORK_ROOT=/data/ra/emuru_historical \
METADATA_CSV=/data/ra/metadata.csv \
bash scripts/prepare_historical_swedish.sh
```

Or run each stage explicitly:

```bash
python -m historical_swedish.extract_page_lines \
  --root /data/ra/pagexml \
  --output /data/ra/emuru_historical/extracted \
  --metadata-csv /data/ra/metadata.csv

python -m historical_swedish.build_splits \
  --input /data/ra/emuru_historical/extracted/lines.jsonl \
  --output /data/ra/emuru_historical/lines_split.jsonl \
  --group-by document_id

python -m historical_swedish.build_pairs \
  --input /data/ra/emuru_historical/lines_split.jsonl \
  --output /data/ra/emuru_historical/pairs.jsonl \
  --image-dir /data/ra/emuru_historical/pairs \
  --pair-by page_id \
  --use normalized

python -m historical_swedish.validate_dataset \
  --input /data/ra/emuru_historical/pairs.jsonl

python -m historical_swedish.make_shards \
  --input /data/ra/emuru_historical/lines_split.jsonl \
  --output /data/ra/emuru_historical/shards_vae \
  --mode vae

python -m historical_swedish.make_shards \
  --input /data/ra/emuru_historical/pairs.jsonl \
  --output /data/ra/emuru_historical/shards_emuru \
  --mode emuru
```

The line manifest includes 64-pixel-high raw and conservatively
background-normalized crops, with a maximum width of 768 pixels. PAGE line
polygons are used as masks, so pixels outside each polygon are placed on a white
background instead of retaining the rectangular bounding-box background.
No aggressive binarization is performed. The pair builder selects a different
style line from the same page (or pseudo-writer), concatenates style and target
images, and sets `json.text` to `style_text + " " + gen_text`. This reproduces
Emuru's inference geometry: a real latent prefix is followed autoregressively by
the requested continuation.

Start with `--pair-by page_id`. After writer-change detection has been validated,
replace page IDs with provenance-constrained pseudo-writer IDs and use
`--pair-by pseudo_writer_id`. Do not globally cluster all lines: cluster within
volumes or short consecutive page windows.

An optional conservative clustering scaffold uses the upstream writer encoder:

```bash
python -m historical_swedish.cluster_writers \
  --input /data/ra/emuru_historical/extracted/lines.jsonl \
  --output /data/ra/emuru_historical/lines_with_writers.jsonl \
  --writer-model pretrained_models/emuru_vae_writer_id \
  --within volume_id \
  --distance-threshold 0.18
```

Tune the distance threshold on manually checked page transitions from several
centuries. The provided value is only a starting point. Re-run
`build_splits.py` on the resulting manifest before pairing; never infer writer
clusters after splitting.

## Splits

The default split is by complete `document_id`, not by line, with deterministic
hashing stratified by century. For the final paper, create additional frozen
manifests for:

- unseen writer within a seen era;
- completely held-out archive/series;
- leave-one-century-out extrapolation.

Never allow the same page, writer cluster, document, or near-duplicate crop on
both sides of a reported train/test boundary.

## Training

First benchmark the released VAE. If adaptation is justified:

```bash
accelerate launch -m historical_swedish.train_historical_vae \
  --train-pattern /data/ra/emuru_historical/shards_vae/train_shards.txt \
  --val-pattern /data/ra/emuru_historical/shards_vae/val_shards.txt \
  --base-vae blowing-up-groundhogs/emuru_vae \
  --output-dir /models/ra-emuru-vae \
  --steps-per-epoch 1000 \
  --epochs 5
```

The default learning rate is deliberately low (`1e-5`). Inspect reconstructed
unseen lines and compare L1, stroke preservation, and downstream HTR CER against
the released VAE before adopting the adapted version.

Then fine-tune the released Emuru checkpoint rather than initializing a new T5:

```bash
accelerate launch -m historical_swedish.train_historical_emuru \
  --historical-train-pattern /data/ra/emuru_historical/shards_emuru/train_shards.txt \
  --historical-val-pattern /data/ra/emuru_historical/shards_emuru/val_shards.txt \
  --replay-train-pattern 'https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000000..000048}.tar' \
  --base-model blowing-up-groundhogs/emuru \
  --vae-path /models/ra-emuru-vae/final \
  --output-dir /models/ra-emuru \
  --historical-probability 0.8 \
  --steps-per-epoch 5000 \
  --epochs 10
```

Omit `--vae-path` for the released-VAE baseline. If Hugging Face is blocked,
replace every model or tar URL with a local mirrored path. The sharder writes
exact local shard lists (`train_shards.txt`, etc.) to avoid fragile guessed
brace ranges.

## Training order for 30,000 pages

1. Smoke-test 1,000–2,000 pages spanning eras.
2. Train the released-VAE baseline with 20% FontSquare replay.
3. Benchmark VAE reconstruction; only then run VAE adaptation.
4. Repeat Emuru fine-tuning with the adapted VAE.
5. Upgrade page-level pairing to validated pseudo-writer pairing.
6. Scale to all pages and freeze the evaluation manifests.
7. Evaluate content CER/WER, writer-embedding similarity/HWD, realism, and HTR
   augmentation on unseen writers.

## Important limitations of this scaffold

- PAGE reading order is irrelevant at line level, but extraction quality is not;
  visually audit every era before scaling.
- A page is only an approximate writer unit. Multi-writer pages inject label
  noise.
- The conservative normalizer may suppress pale ink. Keep the raw crops and run
  a raw-versus-normalized ablation.
- The upstream training scripts initialize models for pretraining. The supplied
  historical trainer instead loads `Emuru.from_pretrained`, which is essential
  for actual fine-tuning.
- Eruku has a different released code/checkpoint interface. The manifests retain
  `style_text`, `gen_text`, `style_prefix_width`, and provenance so they can be
  adapted to Eruku without repeating PAGE extraction and splitting.
