# SmartRxnEmbeddings

Self-supervised transformer embeddings for reaction SMARTS, trained on [RetroRules v3.0](https://retrorules.org/) using a masked language modelling objective.

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jcorreia11/SmartRxnEmbeddings.git
cd SmartRxnEmbeddings
```

**Prediction only** — just `torch` + DVC to pull the model weights:

```bash
uv sync --no-dev --group data
dvc pull   # restore pretrained model weights
```

**Full install** — adds preprocessing, training, evaluation, and visualisation dependencies:

```bash
uv sync --extra full
dvc pull   # restore model weights, embeddings, and processed data
```

The `dev` group (DVC + pytest) is included automatically with `uv sync`.

## Getting started

### Command line

```bash
# Single SMARTS → JSON to stdout
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]"

# Batch from file → numpy array
smarts-embed --file smarts.txt --output embeddings.npy

# Pipe from stdin → CSV
echo "[C:1]-[O:2]>>[C:1]=[O:2]" | smarts-embed --format csv
```

The model checkpoint is auto-discovered from `models/`. To use a specific one:

```bash
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" \
    --weights models/smarts_transformer_20260410_111830.pt \
    --config  models/smarts_transformer_20260410_111830.json
```

Full options:

```
positional args:  one or more SMARTS strings
--file PATH       text file with one SMARTS per line (alternative to positional)
--output PATH     write to file; format inferred from extension (.npy, .json, .csv)
--format          override output format: npy | json | csv  (default: json)
--pooling         cls | mean  (default: cls)
--batch-size N    sequences per forward pass (default: 64)
--max-length N    pad/truncate length (default: from model)
--device          cuda | cpu  (auto-detected)
--weights PATH    model weights (.pt)
--config PATH     model config (.json)
--vocab PATH      vocab.json (default: data/processed/vocab.json)
```

### Python API

```python
from smart_rxn_embeddings.predict import predict, load_embedder

# Single reaction — returns (d_model,) array
emb = predict("[C:1]-[O:2]>>[C:1]=[O:2]")

# Batch — returns (N, d_model) array
embs = predict([
    "[C:1]-[O:2]>>[C:1]=[O:2]",
    "c1ccccc1>>c1cccnc1",
])

# Reuse the same loaded model for multiple calls
embedder = load_embedder(pooling="mean")
embs = embedder.embed(smarts_list, batch_size=128)
```

Both functions auto-discover the latest checkpoint in `models/`. Pass `weights=`, `config=`, and `vocab=` to use a specific run.

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
src/smart_rxn_embeddings/
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

Preprocessing is managed with [DVC](https://dvc.org/):

```bash
dvc repro
```

| Stage | Input | Output |
|---|---|---|
| `load_data` | `data/raw/retrorules-v3.0-*.csv` | `data/processed/clean_smarts.txt` |
| `validate_smarts` | `clean_smarts.txt` | `data/processed/validated_smarts.csv` |
| `build_vocab` | `validated_smarts.csv` | `data/processed/vocab.json` |

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