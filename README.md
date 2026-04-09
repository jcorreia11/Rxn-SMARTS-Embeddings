# SmartRxnEmbeddings

Self-supervised transformer embeddings for reaction SMARTS, trained on the [RetroRules v3.0](https://retrorules.org/) database using a masked-language modelling (MLM) objective.

## Overview

The pipeline takes raw reaction SMARTS from RetroRules, validates them with RDKit, tokenizes them with a rule-based SMARTS tokenizer, and pretrains a BERT-style transformer encoder. The resulting model learns chemical patterns — atom environments, bond contexts, reaction centres — directly from SMARTS syntax without any labels.

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
  SMARTSDataset      tokenize → pad → attention mask
      │
      ▼
  MLMCollator        80/10/10 BERT masking ([MASK] injected at runtime)
      │
      ▼
SmartsMLMModel       TransformerEncoder + MLM head
      │
      ▼
  models/smarts_transformer.pt
```

## Project structure

```
src/smart_rxn_embeddings/
├── preprocessing/
│   ├── load_data.py          # load & deduplicate RetroRules CSVs
│   └── validate_smarts.py    # RDKit validation, extract structural features
├── tokenization/
│   ├── smarts_tokenizer.py   # rule-based tokenizer (Daylight SMARTS grammar)
│   ├── sentencepiece_tokenizer.py
│   └── build_vocab.py        # build frequency-sorted vocab.json
├── datasets/
│   └── smarts_dataset.py     # PyTorch Dataset → (input_ids, attention_mask)
├── models/
│   └── smarts_transformer.py # TransformerConfig, encoder, MLM head
└── training/
    └── mlm_trainer.py        # MLMCollator, TrainingConfig, Trainer

scripts/
├── train_mlm.py              # CLI entry point for MLM pretraining
├── train_mlm.sbatch          # SLURM job script (A100) — auto dvc-tracks model on completion
├── extract_embeddings.py     # extract reaction embeddings from trained model
├── extract_embeddings.sbatch # SLURM job script — auto dvc-tracks embeddings on completion
├── ablation_pooling.py       # CLS vs mean pooling ablation (EC classification)
├── ablation_pooling.sbatch
├── train_ec_classifier.py    # EC number classification benchmark
├── train_ec_classifier.sbatch
├── compare_tokenizers.py     # tokenizer comparison analysis
├── compare_tokenizers.sbatch
├── similarity_correlation.py # embedding vs Tanimoto similarity correlation
├── similarity_correlation.sbatch
├── plot_umap.py              # UMAP visualization of embedding space
├── plot_umap.sbatch
├── nearest_neighbors.py      # nearest-neighbor retrieval in embedding space
├── nearest_neighbors.sbatch
├── plot_dataset_stats.py     # dataset overview figures
├── plot_dataset_stats.sbatch
└── dvc_repro.sbatch          # SLURM job script for DVC preprocessing pipeline

tests/
├── preprocessing/
├── tokenization/
├── datasets/
├── models/
└── training/
```

## Installation

Requires Python ≥ 3.9 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jcorreia11/SmartRxnEmbeddings.git
cd SmartRxnEmbeddings
uv sync
```

## Data pipeline

Data processing is managed with [DVC](https://dvc.org/). To reproduce all preprocessing steps:

```bash
dvc repro
```

This runs (in order):

| Stage | Input | Output |
|---|---|---|
| `load_data` | `data/raw/retrorules-v3.0-*.csv` | `data/processed/clean_smarts.txt` |
| `validate_smarts` | `clean_smarts.txt` | `data/processed/validated_smarts.csv` |
| `build_vocab` | `validated_smarts.csv` | `data/processed/vocab.json` |
| `train_sentencepiece` | `validated_smarts.csv` | `data/processed/sp_tokenizer.model` |

Large binary artifacts (model weights, embeddings) are tracked outside the DVC pipeline using `dvc add`, so they can be versioned and shared without re-running training:

```bash
dvc pull   # restore all tracked artifacts (preprocessed data + model weights + embeddings)
```

## Training

### Locally

```bash
python scripts/train_mlm.py \
    --data  data/processed/validated_smarts.csv \
    --vocab data/processed/vocab.json \
    --epochs 20
```

All options:

```
data:
  --data PATH          CSV with 'smarts' and 'valid' columns
  --vocab PATH         vocab.json produced by build_vocab.py
  --max-length INT     pad/truncate sequences to this length (default: 256)

model:
  --d-model INT        embedding dimension (default: 256)
  --nhead INT          attention heads (default: 8)
  --num-layers INT     encoder layers (default: 6)
  --dim-feedforward INT  FFN hidden dim (default: 1024)
  --dropout FLOAT      dropout probability (default: 0.1)

training:
  --lr FLOAT           learning rate (default: 1e-4)
  --batch-size INT     batch size (default: 128)
  --epochs INT         training epochs (default: 40)
  --mask-prob FLOAT    fraction of tokens masked (default: 0.15)
  --val-split FLOAT    validation fraction (default: 0.1)
  --warmup-steps INT   linear LR warmup steps (default: 500)
  --max-grad-norm FLOAT  gradient clipping norm (default: 1.0)
  --num-workers INT    DataLoader workers (default: 4)
  --checkpoint-dir DIR best checkpoint directory (default: models/checkpoints)
  --output PATH        final model weights (default: models/smarts_transformer_<RUN_ID>.pt)
  --device STR         'cuda' or 'cpu' (auto-detected if omitted)
```

### On HPC (SLURM)

```bash
sbatch scripts/train_mlm.sbatch
```

All hyperparameters are overridable at submission time:

```bash
sbatch --export=ALL,EPOCHS=50,D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,BATCH_SIZE=256 \
    scripts/train_mlm.sbatch
```

Output files (stamped with `RUN_ID=YYYYMMDD_HHMMSS`):

| File | Description |
|---|---|
| `models/smarts_transformer_<RUN_ID>.pt` | Final model weights (DVC-tracked) |
| `models/smarts_transformer_<RUN_ID>.json` | Model + training config (DVC-tracked) |
| `models/checkpoints/<RUN_ID>/best.pt` | Best epoch checkpoint |
| `models/smarts_transformer_<RUN_ID>_training_loss.pdf` | Training/validation loss curves |
| `models/smarts_transformer_<RUN_ID>_report.md` | Run report with config and environment |

The sbatch script automatically runs `dvc add` + `dvc push` on the `.pt` and `.json` files upon completion.

For the full HPC submission workflow (all phases, ordering, and publication-level overrides) see [`experiments-dependency-graph.md`](experiments-dependency-graph.md).

### Loading a trained model

```python
import json
import torch
from smart_rxn_embeddings.models.smarts_transformer import SmartsMLMModel, TransformerConfig

cfg_dict = json.loads(open("models/smarts_transformer_<RUN_ID>.json").read())
model = SmartsMLMModel(TransformerConfig.from_dict(cfg_dict["model_config"]))
model.load_state_dict(torch.load("models/smarts_transformer_<RUN_ID>.pt", weights_only=True))
model.eval()
```

## Masking strategy

`[MASK]` is injected at collator level — existing `vocab.json` files are not modified.
The collator follows the original BERT 80/10/10 rule:

- **80 %** of selected tokens → `[MASK]`
- **10 %** → random token from the vocabulary
- **10 %** → kept unchanged

Only real (non-padding) tokens are eligible for masking.

## Development

```bash
# run all tests
uv run pytest

# lint + format
ruff check src tests scripts
ruff format src tests scripts
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training |
| `rdkit` | SMARTS validation |
| `pandas` | Data loading |
| `sentencepiece` | Alternative tokenizer |
| `dvc` | Data pipeline reproducibility |