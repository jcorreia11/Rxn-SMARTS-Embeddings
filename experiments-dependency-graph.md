# HPC Submission Guide

## Dependency Graph

```
[dvc_repro]                                    <- Preprocessing (CPU, run once)
     |
     +-------------------------+
     |                         |
[plot_dataset_stats]    [train_mlm]            <- Phase 0 (CPU) and Phase 1 (GPU) share preprocessed data
[compare_tokenizers]         |
                             +----------------+
                             |                |
                  [extract_embeddings]  [ablation_pooling]   <- Phase 2: GPU (parallel)
                             |
             +---------------+---------------+
             |               |               |
       [plot_umap]  [similarity_corr]  [train_ec_classifier] <- Phase 3: CPU (parallel)
                                       (also needs WEIGHTS)
                                             |
                                   [nearest_neighbors]        <- Phase 4: optional
```

## Artifact Persistence (automatic on completion)

| Script | Outputs | How |
|---|---|---|
| `train_mlm` | `models/*.pt` + `*.json` | `dvc add` + `dvc push` |
| `extract_embeddings` | `data/embeddings/*.npy` + `*.txt` | `dvc add` + `dvc push` |
| all others | results JSONs, reports, figures | `git commit` |

---

## Phase 0 -- Independent (submit anytime, parallel, CPU)

No model needed. Run before or alongside Phase 1.

```bash
sbatch scripts/plot_dataset_stats.sbatch
sbatch scripts/compare_tokenizers.sbatch
```

---

## Phase 1 -- Train MLM (GPU)

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
Last command run:

```
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=256,D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4 scripts/train_mlm.sbatch
```

> After completion, note the `RUN_ID` from the job log (`Run ID: YYYYMMDD_HHMMSS`) and set:
> ```bash
> export WEIGHTS=models/smarts_transformer_${RUN_ID}.pt
> export CONFIG=models/smarts_transformer_${RUN_ID}.json
> ```

---

## Phase 2 -- Extract Embeddings + Pooling Ablation (parallel, GPU)

Submit both simultaneously after Phase 1.

```bash
# 2a -- extract embeddings (default: CLS pooling)
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
POOLING=cls,\
BATCH_SIZE=256,\
OUTPUT=data/embeddings/reaction_embeddings.npy,\
SMARTS_OUTPUT=data/embeddings/reaction_smarts.txt \
scripts/extract_embeddings.sbatch

# 2b -- CLS vs mean pooling ablation
# SPLIT_STRATEGY=group (default) keeps RetroRules radius-siblings out of
# both sides of the split -- see the "Train/test split fix" note in
# results/run_log.md. Omit for the default, or set =random only to
# reproduce the old leaky split for a before/after comparison.
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
N_SAMPLES=20000,\
EC_DEPTH=1,\
FOLDS=10,\
EMBED_BATCH_SIZE=128,\
SPLIT_STRATEGY=group \
scripts/ablation_pooling.sbatch
```

> Once 2b finishes, check the report for the winning pooling strategy, then set:
> ```bash
> export BEST_POOLING=cls   # or mean, whichever won
> ```
> If the winner differs from what was used in 2a, re-run `extract_embeddings.sbatch` with `POOLING=${BEST_POOLING}` before submitting Phase 3.

---

## Phase 3 -- Evaluation (parallel, CPU)

Submit all three simultaneously after 2a completes.

```bash
# 3a -- UMAP visualization
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

# 3b -- Embedding vs Tanimoto similarity correlation
#       N_REACTIONS=3000 gives ~4.5M pairs, statistically robust, fits in 32 GB
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/reaction_embeddings.npy,\
SMARTS_FILE=data/embeddings/reaction_smarts.txt,\
N_REACTIONS=3000,\
SEED=42,\
FORMAT=pdf,\
DPI=300,\
GRIDSIZE=60 \
scripts/similarity_correlation.sbatch

# 3c -- EC classification benchmark (TF-IDF vs pretrained vs random embeddings)
# Set POOLING to the winner from ablation_pooling (2b) — default in script is mean
# SPLIT_STRATEGY=group (default) — see note on 2b above.
sbatch --export=ALL,\
WEIGHTS=${WEIGHTS},\
CONFIG=${CONFIG},\
POOLING=${BEST_POOLING},\
N_SAMPLES=50000,\
EC_DEPTH=1,\
FOLDS=10,\
EMBED_BATCH_SIZE=256,\
SPLIT_STRATEGY=group \
scripts/train_ec_classifier.sbatch
```

---

## Phase 4 -- Nearest Neighbors (optional, qualitative)

Lightweight CPU job. Submit after Phase 2.

```bash
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
| `train_mlm` | `EPOCHS` | 40 | **100** | Ensure convergence |
| `train_mlm` | `WARMUP_STEPS` | 500 | **2000** | Large dataset needs longer warmup |
| `train_mlm` | `BATCH_SIZE` | 128 | **64** | Better gradient signal per step |
| `ablation_pooling` | `N_SAMPLES` | 10000 | **20000** | More reliable CV estimate |
| `ablation_pooling` | `FOLDS` | 5 | **10** | Tighter confidence intervals |
| `train_ec_classifier` | `N_SAMPLES` | 10000 | **50000+** | Use as much data as available |
| `train_ec_classifier` | `FOLDS` | 5 | **10** | Publication standard |
| `similarity_correlation` | `N_REACTIONS` | 1000 | **3000** | Default gives only ~500k pairs -- too small |

---

## Preprocessing -- `dvc_repro.sbatch` (run once, CPU)

Runs the full preprocessing chain via `dvc repro`:

```
raw CSVs -> load_data -> validate_smarts -> build_vocab -> train_sentencepiece
```

```bash
sbatch scripts/dvc_repro.sbatch
```

**One-time HPC remote setup (before first `dvc push`):**
```bash
dvc remote add --default hpc_storage /projects/F202508983CPCAA0/jcorreia/dvc-storage
git add .dvc/config
git commit -m "[DVC] Set HPC remote storage"
```