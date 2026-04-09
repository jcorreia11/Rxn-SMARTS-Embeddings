# HPC Submission Guide

## Dependency Graph

```
[plot_dataset_stats]  [compare_tokenizers]     ← Phase 0: independent, paper figures
         |                    |
         └────── (no deps) ───┘

[train_mlm]                                    ← Phase 1: core training (GPU required)
         |
         ├──── [extract_embeddings]             ← Phase 2a: parallel
         └──── [ablation_pooling]               ← Phase 2b: parallel
                      |
          ┌───────────┼─────────────┐
   [plot_umap]  [similarity_corr]  [train_ec_classifier]  ← Phase 3: parallel (after embeddings)
                                          |
                              [nearest_neighbors]          ← Phase 4: optional/qualitative
```

## Phase 0 — Independent (submit anytime, parallel)

No model needed. Pure data analysis / paper figures.

```bash
sbatch scripts/plot_dataset_stats.sbatch
sbatch scripts/compare_tokenizers.sbatch
```

---

## Phase 1 — Train MLM (sequential, GPU)

Publication-level overrides: more epochs, more warmup (larger dataset needs longer schedule), slightly tighter gradient clipping.

```bash
sbatch --export=ALL,\
EPOCHS=100,\
WARMUP_STEPS=2000,\
BATCH_SIZE=64,\
LR=1e-4,\
MAX_LENGTH=256,\
D_MODEL=256,\
NHEAD=8,\
NUM_LAYERS=6,\
DIM_FEEDFORWARD=1024,\
DROPOUT=0.1,\
MASK_PROB=0.15,\
VAL_SPLIT=0.1,\
MAX_GRAD_NORM=1.0,\
NUM_WORKERS=4 \
scripts/train_mlm.sbatch
```

> After this completes, note the `RUN_ID` from the job log (printed as `Run ID: YYYYMMDD_HHMMSS`) and set:
> ```bash
> export RUN_ID=<value_from_log>
> export WEIGHTS=models/smarts_transformer_${RUN_ID}.pt
> export CONFIG=models/smarts_transformer_${RUN_ID}.json
> ```

---

## Phase 2 — Extract Embeddings + Pooling Ablation (parallel, both GPU)

Submit simultaneously after Phase 1.

```bash
# 2a — extract embeddings (use mean pooling if ablation hasn't run yet;
#       re-run with winning strategy after ablation — but cls is the default)
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
POOLING=cls,\
BATCH_SIZE=256,\
OUTPUT=data/embeddings/reaction_embeddings.npy,\
SMARTS_OUTPUT=data/embeddings/reaction_smarts.txt \
scripts/extract_embeddings.sbatch

# 2b — CLS vs mean pooling ablation
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
N_SAMPLES=20000,\
EC_DEPTH=1,\
FOLDS=10,\
EMBED_BATCH_SIZE=128 \
scripts/ablation_pooling.sbatch
```

> Once 2b finishes, check which pooling wins and re-run 2a with the winning `POOLING` if needed before Phase 3.

---

## Phase 3 — Evaluation (parallel, after embeddings are ready)

Submit all three simultaneously.

```bash
# 3a — UMAP visualization (GPU node for speed, but mostly CPU-bound)
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/reaction_embeddings.npy,\
SMARTS_FILE=data/embeddings/reaction_smarts.txt,\
N_NEIGHBORS=15,\
MIN_DIST=0.1,\
METRIC=cosine,\
RANDOM_STATE=42,\
DPI=300,\
FORMAT=pdf \
scripts/plot_umap.sbatch

# 3b — Embedding vs Tanimoto similarity correlation
#       N_REACTIONS=3000 → ~4.5M pairs, statistically robust, still fits in 32GB
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/reaction_embeddings.npy,\
SMARTS_FILE=data/embeddings/reaction_smarts.txt,\
N_REACTIONS=3000,\
SEED=42,\
FORMAT=pdf,\
DPI=300,\
GRIDSIZE=60 \
scripts/similarity_correlation.sbatch

# 3c — EC classification benchmark (TF-IDF vs pretrained vs random embeddings)
#       N_SAMPLES=-1 or large to use all available data; 10-fold CV for publication
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
VOCAB=data/processed/vocab.json,\
POOLING=cls,\
N_SAMPLES=50000,\
EC_DEPTH=1,\
FOLDS=10,\
EMBED_BATCH_SIZE=256 \
scripts/train_ec_classifier.sbatch
```

---

## Phase 4 — Nearest Neighbors (optional / qualitative examples)

Lightweight, CPU-only node. Submit after Phase 2.

```bash
# A few illustrative queries for the paper
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/reaction_embeddings.npy,\
SMARTS_FILE=data/embeddings/reaction_smarts.txt,\
TOP_K=10,\
METRIC=cosine,\
QUERY_MODE=random,\
SEED=42 \
scripts/nearest_neighbors.sbatch
```

---

## Key Publication-Level Argument Changes vs. Defaults

| Script | Arg | Default | Recommended | Reason |
|---|---|---|---|---|
| `train_mlm` | `EPOCHS` | 40 | 100 | Ensure convergence |
| `train_mlm` | `WARMUP_STEPS` | 500 | 2000 | Large dataset needs longer warmup |
| `train_mlm` | `BATCH_SIZE` | 128 | 64 | Better gradient signal per step |
| `ablation_pooling` | `N_SAMPLES` | 10000 | 20000 | More reliable CV estimate |
| `ablation_pooling` | `FOLDS` | 5 | 10 | Tighter confidence intervals |
| `train_ec_classifier` | `N_SAMPLES` | 10000 | 50000+ | Use as much data as available |
| `train_ec_classifier` | `FOLDS` | 5 | 10 | Publication standard |
| `similarity_correlation` | `N_REACTIONS` | 1000 | 3000 | Default gives only ~500k pairs — too small |

---

## Note on `dvc_repro.sbatch`

If your DVC pipeline is fully configured to orchestrate the stages in the right order, `dvc_repro.sbatch` can run the entire preprocessing pipeline in one shot. However, it runs on CPU-only (`normal-x86`) and does not handle the GPU training scripts — those must still be submitted manually as above. Use it for data preprocessing stages if they're DVC-tracked.
