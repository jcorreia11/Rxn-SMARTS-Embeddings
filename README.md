# Rxn-SMARTS-Embeddings

[![CI](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/rxn-smarts-embeddings/badge/?version=latest)](https://rxn-smarts-embeddings.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Self-supervised transformer embeddings for reaction SMARTS, pretrained via
span-masked language modelling directly on [RetroRules v3.0](https://retrorules.org/).

- **Paper**: citation to follow upon publication
- **Companion repo** (experiment pipeline, SLURM scripts, paper stats): [Rxn-SMARTS-Embeddings-paper](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings-paper)
- **Pretrained weights**: [Hugging Face Hub](https://huggingface.co/jcorreia11/Rxn-SMARTS-Embeddings)
- **Training corpus, tokenizer, embeddings**: [Zenodo](https://doi.org/10.5281/zenodo.22645328)
- **Full experiment results/logs**: [Zenodo](https://doi.org/10.5281/zenodo.22646105)

## Installation

Requires Python 3.10–3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jcorreia11/Rxn-SMARTS-Embeddings.git
cd Rxn-SMARTS-Embeddings

uv sync --no-dev          # embedding SMARTS only — just torch + numpy
uv sync --extra all       # + preprocessing/training/evaluation/visualisation, for the full pipeline
```

## Getting started

No setup beyond installation is required — the first call downloads and caches
the medium model and vocabulary from the Hugging Face Hub automatically.

```bash
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]"                      # → JSON to stdout
smarts-embed --file smarts.txt --output embeddings.npy       # batch → numpy array
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" --size large          # small | medium | large
```

```python
from rxn_smarts_embeddings.predict import predict, load_embedder

emb = predict("[C:1]-[O:2]>>[C:1]=[O:2]")          # (d_model,)
embs = predict(["[C:1]-[O:2]>>[C:1]=[O:2]", "c1ccccc1>>c1cccnc1"])  # (N, d_model)

embedder = load_embedder()                          # reuse across calls
embs = embedder.embed(smarts_list, batch_size=128)
```

Both auto-discover the latest local checkpoint in `models/`, falling back to a
Hugging Face Hub download when none is found. Pass `weights=`/`config=`/`vocab=`
(or `--weights`/`--config`/`--vocab`) to use a specific run instead.

Run `smarts-embed --help` for the full option list (pooling, batch size, output
format, device, etc.).

## Project structure

```
src/rxn_smarts_embeddings/
├── predict.py       # predict() API + smarts-embed CLI entry point
├── preprocessing/   # load & deduplicate RetroRules CSVs, RDKit validation
├── tokenization/    # rule-based tokenizer (Daylight SMARTS grammar) + vocab
├── datasets/        # SMARTSDataset (tokenize, pad, batch)
├── models/          # TransformerConfig, encoder, MLM head, SmartsEmbedder
└── training/        # MLMCollator, Trainer
```

## Reproducing the data pipeline

Download the raw RetroRules v3.0 CSVs from [retrorules.org](https://retrorules.org/)
into `data/raw/`, then:

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

Training uses `Trainer` (`rxn_smarts_embeddings.training.mlm_trainer`); output
checkpoints are stamped `models/smarts_transformer_<YYYYMMDD_HHMMSS>.{pt,json}`.
For the CLI wrapper, SLURM `.sbatch` scripts, and full HPC reproduction guide, see
the [companion repo](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings-paper).

## Development

```bash
uv run pytest
ruff check src tests && ruff format src tests
```

Docs (built with [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/)):

```bash
uv sync --group docs
uv run mkdocs serve   # http://127.0.0.1:8000, live-reloads on edit
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

Citation details will be added once the paper is published.