# Rxn-SMARTS-Embeddings

Self-supervised transformer embeddings for reaction SMARTS, trained on [RetroRules v3.0](https://retrorules.org/) using a masked language modelling objective.

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jcorreia11/Rxn-SMARTS-Embeddings.git
cd Rxn-SMARTS-Embeddings
```

**Embedding SMARTS (default)** — just `torch` + `numpy`:

```bash
uv sync --no-dev
```

**Everything** — adds preprocessing, training, evaluation, and visualisation dependencies, for reproducing the full experiment pipeline:

```bash
uv sync --extra all
```

The `dev` group (pytest) is included automatically with `uv sync`.

## Getting started

No setup beyond installation is required: the first call downloads and caches
the medium model and vocabulary from the
[Hugging Face Hub](https://huggingface.co/jcorreia11/Rxn-SMARTS-Embeddings)
automatically. If a local checkpoint already exists in `models/` (e.g. from
training your own), that one is used instead — see "Data pipeline" and
"Training" below to reproduce one.

### Command line

```bash
# Single SMARTS → JSON to stdout
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]"

# Batch from file → numpy array
smarts-embed --file smarts.txt --output embeddings.npy

# Pipe from stdin → CSV
echo "[C:1]-[O:2]>>[C:1]=[O:2]" | smarts-embed --format csv

# Use the small or large released model instead of medium
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" --size large
```

To use a specific local checkpoint instead of the auto-discovered/downloaded one:

```bash
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" \
    --weights models/smarts_transformer_20260715_111009.pt \
    --config  models/smarts_transformer_20260715_111009.json
```

Full options:

```
positional args:  one or more SMARTS strings
--file PATH       text file with one SMARTS per line (alternative to positional)
--output PATH     write to file; format inferred from extension (.npy, .json, .csv)
--format          override output format: npy | json | csv  (default: json)
--pooling         mean | cls  (default: mean)
--batch-size N    sequences per forward pass (default: 64)
--max-length N    pad/truncate length (default: from model)
--device          cuda | cpu  (auto-detected)
--weights PATH    model weights (.pt); auto-discovered locally, else downloaded
--config PATH     model config (.json)
--vocab PATH      vocab.json (default: data/processed/vocab.json, else downloaded)
--size            small | medium | large  (default: medium) — which released
                  model to download when no local checkpoint is found
```

### Python API

```python
from rxn_smarts_embeddings.predict import predict, load_embedder

# Single reaction — returns (d_model,) array
emb = predict("[C:1]-[O:2]>>[C:1]=[O:2]")

# Batch — returns (N, d_model) array
embs = predict([
    "[C:1]-[O:2]>>[C:1]=[O:2]",
    "c1ccccc1>>c1cccnc1",
])

# Reuse the same loaded model for multiple calls
embedder = load_embedder()
embs = embedder.embed(smarts_list, batch_size=128)
```

Both functions auto-discover the latest local checkpoint in `models/`, falling
back to a Hugging Face Hub download (`hf_size="medium"` by default) when none
is found. Pass `weights=`, `config=`, and `vocab=` to use a specific run.

## Overview

The pipeline trains a BERT-style transformer encoder on reaction SMARTS using span-masked language modelling. The model learns chemical patterns — atom environments, bond contexts, reaction centres — directly from SMARTS syntax without labels.

```
RetroRules CSV
      │
      ▼
  load_data          clean & deduplicate
      │
      ▼
 validate_smarts     RDKit validation + feature extraction
      │
      ▼
  build_vocab        rule-based tokenization → vocab.json
      │
      ▼
SmartsMLMModel       TransformerEncoder (d_model=256, 6 layers, 8 heads)
      │
      ▼
  models/smarts_transformer_<RUN_ID>.pt
```

## Project structure

```
src/rxn_smarts_embeddings/
├── predict.py            # predict() API + smarts-embed CLI entry point
├── preprocessing/
│   ├── load_data.py      # load & deduplicate RetroRules CSVs
│   └── validate_smarts.py
├── tokenization/
│   ├── smarts_tokenizer.py   # rule-based tokenizer (Daylight SMARTS grammar)
│   └── build_vocab.py
├── datasets/
│   └── smarts_dataset.py
├── models/
│   ├── smarts_transformer.py # TransformerConfig, encoder, MLM head
│   └── embed.py              # SmartsEmbedder
└── training/
    └── mlm_trainer.py        # MLMCollator, Trainer

scripts/
├── train_mlm.py              # MLM pretraining
├── extract_embeddings.py     # batch embedding extraction from a validated CSV
└── train_ec_classifier.py    # EC number classification benchmark
```

## Data pipeline

Preprocessing is a plain sequence of scripts — each is cheap (a couple of
minutes on the full corpus) and deterministic, so just re-run them in order
when a dependency changes:

```bash
python src/rxn_smarts_embeddings/preprocessing/load_data.py
python src/rxn_smarts_embeddings/preprocessing/validate_smarts.py
python src/rxn_smarts_embeddings/tokenization/build_vocab.py
python src/rxn_smarts_embeddings/tokenization/sentencepiece_tokenizer.py
```

| Stage | Input | Output |
|---|---|---|
| `load_data.py` | `data/raw/retrorules-v3.0-*.csv` | `data/processed/clean_smarts.txt`, `reaction_groups.csv` |
| `validate_smarts.py` | `clean_smarts.txt`, `reaction_groups.csv` | `data/processed/validated_smarts.csv` |
| `build_vocab.py` | `validated_smarts.csv` | `data/processed/vocab.json` |
| `sentencepiece_tokenizer.py` | `validated_smarts.csv` | `data/processed/sp_tokenizer.{model,vocab}` |

On HPC, submit the four commands above directly, or wrap them in your own job script (see `scripts/preprocess.sbatch`).

## Training

```bash
python scripts/train_mlm.py \
    --data  data/processed/validated_smarts.csv \
    --vocab data/processed/vocab.json \
    --epochs 100 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
    --batch-size 64 --val-split 0.1 --warmup-steps 2000
```

On HPC (SLURM / A100):

```bash
sbatch scripts/train_mlm.sbatch

# Override hyperparameters at submission time
sbatch --export=ALL,EPOCHS=100,D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,BATCH_SIZE=256 \
    scripts/train_mlm.sbatch
```

Output files are stamped `models/smarts_transformer_<YYYYMMDD_HHMMSS>.{pt,json}`.

## Development

```bash
uv run pytest
ruff check src tests scripts && ruff format src tests scripts
```