# Rxn-SMARTS-Embeddings

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

No setup beyond installation is required — the first call downloads and caches
the medium model and vocabulary from the Hugging Face Hub automatically. See
[Quick Start](quickstart.md) to get going, or the [API Reference](api.md) for
the full package.

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

Reproducing the data pipeline, training a model from scratch, and the SLURM
job scripts used for the paper's experiments are documented in the
[main repo's README](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings#readme)
and the [companion repo](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings-paper).

## License

MIT — see [LICENSE](https://github.com/jcorreia11/Rxn-SMARTS-Embeddings/blob/main/LICENSE).