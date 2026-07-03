# SmartRxnEmbeddings — Experiment Runbook

## Models

| Name   | d_model | nhead | Layers | FFN  | max_seq_len | Batch | Status                        |
|--------|---------|-------|--------|------|-------------|-------|-------------------------------|
| small  | 256     | 8     | 6      | 1024 | 256         | 64    | Done (`20260410_111830`)      |
| medium | 256     | 8     | 6      | 1024 | 512         | 64    | Done (`20260428_160310`)      |
| large  | 512     | 16    | 8      | 2048 | 512         | 32    | Running (`20260513_093030`)   |

---

## Proposed Directory Structure

```
models/
  small/
    weights.pt
    config.json
    env.json
    training_report.md
    training_loss.pdf
    checkpoints/
  medium/  (same layout)
  large/   (same layout)

data/embeddings/
  small/
    embeddings.npy
    smarts.txt
  medium/  (same layout)
  large/   (same layout)

results/
  small/
    pooling_ablation/
      results.json  env.json  report.md  report.pdf
    ec_classifier/
      depth1/  results.json  env.json  report.md  report.pdf
      depth2/  ...
      depth3/  ...
    similarity_correlation/
      results.json  env.json  report.md  figure.pdf
    nearest_neighbors/
      results.json  env.json  report.md
    umap/
      depth1.pdf  depth2.pdf  depth3.pdf
  medium/  (same layout)
  large/   (same layout)

results/figures/          ← shared UMAP coord caches (intermediate artifacts)
  small_umap_coords.npy
  medium_umap_coords.npy
  large_umap_coords.npy
```

---

## One-time Script Fixes

Two scripts hardcode output paths. Add `:-` defaults so they accept `--export` overrides.

**`scripts/train_mlm.sbatch`** — around the `CHECKPOINT_DIR` / `OUTPUT` assignments:
```bash
# Before
CHECKPOINT_DIR="models/checkpoints/${RUN_ID}"
OUTPUT="models/smarts_transformer_${RUN_ID}.pt"

# After
CHECKPOINT_DIR="${CHECKPOINT_DIR:-models/checkpoints/${RUN_ID}}"
OUTPUT="${OUTPUT:-models/smarts_transformer_${RUN_ID}.pt}"
```

**`scripts/plot_umap.sbatch`** — the `UMAP_CACHE` assignment:
```bash
# Before
UMAP_CACHE="results/figures/${EMBEDDINGS_STEM}_umap_coords.npy"

# After
UMAP_CACHE="${UMAP_CACHE:-results/figures/${EMBEDDINGS_STEM}_umap_coords.npy}"
```

**`scripts/train_ec_classifier.sbatch`** — around the `OUTPUT` / `ENV_JSON` / `REPORT` assignments:
```bash
# Before
OUTPUT="results/ec_classifier_${RUN_ID}.json"
ENV_JSON="results/ec_classifier_${RUN_ID}_env.json"
REPORT="results/ec_classifier_${RUN_ID}_report.md"

# After
OUTPUT="${OUTPUT:-results/ec_classifier_${RUN_ID}.json}"
ENV_JSON="${ENV_JSON:-results/ec_classifier_${RUN_ID}_env.json}"
REPORT="${REPORT:-results/ec_classifier_${RUN_ID}_report.md}"
```

---

## Execution Plan

```
Phase 1 ── Train MLM ─────────────── small (done), medium (done), large (running)
                │
                ▼  (per model, after training finishes)
Phase 2 ── Extract Embeddings ─────── small ║ medium ║ large  (parallel, ~3 min each)
                │
       ┌────────┴──────────────────────────────────────────────────┐
       ▼                                                           ▼
Phase 3a ── Pooling Ablation ──────── small ║ medium ║ large  (parallel, ~47 min each)
Phase 3b ── EC Classifier ─────────── 9 jobs total (3 models × 3 depths), all parallel
Phase 3c ── Similarity Correlation ── small ║ medium ║ large  (parallel, ~1 min each)
Phase 3c-random ── Random-init Baseline ── all sizes in one job (~5 min total)
Phase 3d ── Nearest Neighbors ──────── small ║ medium ║ large  (parallel, <1 min each)
Phase 3e ── UMAP depth=1 ──────────── small ║ medium ║ large  (parallel, ~13 min each)
                │
                ▼  (reuses cached UMAP coords, runs fast)
Phase 3f ── UMAP depth=2,3 ────────── 6 jobs (3 models × 2 depths), all parallel (~30 sec each)
```

---

## Phase 1 — Train MLM (GPU A100, ~3.5 h / 6 h / unknown)

Small and medium are already trained. Only large needs to run.
The large model is currently running as job `1257419` (run ID `20260513_093030`).

If you need to re-run any of them:

```bash
# small (d_model=256, max_seq_len=256)
sbatch --export=ALL,\
EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=256,\
D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,\
DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,\
VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,\
CHECKPOINT_DIR=models/small/checkpoints,\
OUTPUT=models/small/weights.pt \
scripts/train_mlm.sbatch

# medium (d_model=256, max_seq_len=512)
sbatch --export=ALL,\
EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=512,\
D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,\
DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,\
VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,\
CHECKPOINT_DIR=models/medium/checkpoints,\
OUTPUT=models/medium/weights.pt \
scripts/train_mlm.sbatch

# large (d_model=512, max_seq_len=512)
sbatch --export=ALL,\
EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=32,LR=1e-4,MAX_LENGTH=512,\
D_MODEL=512,NHEAD=16,NUM_LAYERS=8,DIM_FEEDFORWARD=2048,\
DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,\
VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,\
CHECKPOINT_DIR=models/large/checkpoints,\
OUTPUT=models/large/weights.pt \
scripts/train_mlm.sbatch
```

> **Existing weights** (no re-training needed unless you want the clean directory layout):
> - small  → `models/smarts_transformer_20260410_111830.pt`
> - medium → `models/smarts_transformer_20260428_160310.pt`
> - large  → `models/smarts_transformer_20260513_093030.pt` (once job 1257419 finishes)

---

## Phase 2 — Extract Embeddings (GPU A100, ~3 min each)

All three can be submitted at the same time. Run from `scripts/`.

```bash
# small
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260410_111830.pt,\
CONFIG=models/smarts_transformer_20260410_111830.json,\
POOLING=mean,BATCH_SIZE=256,\
OUTPUT=data/embeddings/small/embeddings.npy,\
SMARTS_OUTPUT=data/embeddings/small/smarts.txt \
extract_embeddings.sbatch

# medium
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260428_160310.pt,\
CONFIG=models/smarts_transformer_20260428_160310.json,\
POOLING=mean,BATCH_SIZE=256,\
OUTPUT=data/embeddings/medium/embeddings.npy,\
SMARTS_OUTPUT=data/embeddings/medium/smarts.txt \
extract_embeddings.sbatch

# large  (wait for Phase 1 large to finish first)
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260513_093030.pt,\
CONFIG=models/smarts_transformer_20260513_093030.json,\
POOLING=mean,BATCH_SIZE=256,\
OUTPUT=data/embeddings/large/embeddings.npy,\
SMARTS_OUTPUT=data/embeddings/large/smarts.txt \
extract_embeddings.sbatch
```

---

## Phase 3a — Pooling Ablation (GPU A100, ~47 min each)

Run at **EC depth=1 only** for all models. The pooling strategy (CLS vs mean) is a property of
the embeddings themselves, not of the downstream task — if mean wins at depth=1 it will win at
all depths. Only revisit other depths if depth=1 gives an ambiguous result.

All three can run in parallel. Run from `scripts/`.

```bash
# small
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260410_111830.pt,\
CONFIG=models/smarts_transformer_20260410_111830.json,\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,\
OUTPUT_JSON=results/small/pooling_ablation/results.json,\
OUTPUT_ENV=results/small/pooling_ablation/env.json,\
OUTPUT_REPORT=results/small/pooling_ablation/report.md \
ablation_pooling.sbatch

# medium
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260428_160310.pt,\
CONFIG=models/smarts_transformer_20260428_160310.json,\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,\
OUTPUT_JSON=results/medium/pooling_ablation/results.json,\
OUTPUT_ENV=results/medium/pooling_ablation/env.json,\
OUTPUT_REPORT=results/medium/pooling_ablation/report.md \
ablation_pooling.sbatch

# large  (wait for Phase 1 large to finish first)
sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260513_093030.pt,\
CONFIG=models/smarts_transformer_20260513_093030.json,\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,\
OUTPUT_JSON=results/large/pooling_ablation/results.json,\
OUTPUT_ENV=results/large/pooling_ablation/env.json,\
OUTPUT_REPORT=results/large/pooling_ablation/report.md \
ablation_pooling.sbatch
```

---

## Phase 3b — EC Classifier (CPU/GPU, ~13–50 min each)

All 9 jobs can run in parallel once embeddings are extracted (Phase 2).
Run from `scripts/`.

```bash
# ── small ──────────────────────────────────────────────────────────────────

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260410_111830.pt,\
CONFIG=models/smarts_transformer_20260410_111830.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=1,\
OUTPUT=results/small/ec_classifier/depth1/results.json,\
ENV_JSON=results/small/ec_classifier/depth1/env.json,\
REPORT=results/small/ec_classifier/depth1/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260410_111830.pt,\
CONFIG=models/smarts_transformer_20260410_111830.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=2,\
OUTPUT=results/small/ec_classifier/depth2/results.json,\
ENV_JSON=results/small/ec_classifier/depth2/env.json,\
REPORT=results/small/ec_classifier/depth2/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260410_111830.pt,\
CONFIG=models/smarts_transformer_20260410_111830.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=3,\
OUTPUT=results/small/ec_classifier/depth3/results.json,\
ENV_JSON=results/small/ec_classifier/depth3/env.json,\
REPORT=results/small/ec_classifier/depth3/report.md \
train_ec_classifier.sbatch

# ── medium ─────────────────────────────────────────────────────────────────

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260428_160310.pt,\
CONFIG=models/smarts_transformer_20260428_160310.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=1,\
OUTPUT=results/medium/ec_classifier/depth1/results.json,\
ENV_JSON=results/medium/ec_classifier/depth1/env.json,\
REPORT=results/medium/ec_classifier/depth1/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260428_160310.pt,\
CONFIG=models/smarts_transformer_20260428_160310.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=2,\
OUTPUT=results/medium/ec_classifier/depth2/results.json,\
ENV_JSON=results/medium/ec_classifier/depth2/env.json,\
REPORT=results/medium/ec_classifier/depth2/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260428_160310.pt,\
CONFIG=models/smarts_transformer_20260428_160310.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=3,\
OUTPUT=results/medium/ec_classifier/depth3/results.json,\
ENV_JSON=results/medium/ec_classifier/depth3/env.json,\
REPORT=results/medium/ec_classifier/depth3/report.md \
train_ec_classifier.sbatch

# ── large  (wait for Phase 2 large to finish first) ───────────────────────

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260513_093030.pt,\
CONFIG=models/smarts_transformer_20260513_093030.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=1,\
OUTPUT=results/large/ec_classifier/depth1/results.json,\
ENV_JSON=results/large/ec_classifier/depth1/env.json,\
REPORT=results/large/ec_classifier/depth1/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260513_093030.pt,\
CONFIG=models/smarts_transformer_20260513_093030.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=2,\
OUTPUT=results/large/ec_classifier/depth2/results.json,\
ENV_JSON=results/large/ec_classifier/depth2/env.json,\
REPORT=results/large/ec_classifier/depth2/report.md \
train_ec_classifier.sbatch

sbatch --export=ALL,\
WEIGHTS=models/smarts_transformer_20260513_093030.pt,\
CONFIG=models/smarts_transformer_20260513_093030.json,\
POOLING=mean,EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=3,\
OUTPUT=results/large/ec_classifier/depth3/results.json,\
ENV_JSON=results/large/ec_classifier/depth3/env.json,\
REPORT=results/large/ec_classifier/depth3/report.md \
train_ec_classifier.sbatch
```

---

## Phase 3c — Similarity Correlation (CPU, ~1 min each)

All three can run in parallel. Run from `scripts/`.

```bash
# small
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/small/embeddings.npy,\
SMARTS_FILE=data/embeddings/small/smarts.txt,\
N_REACTIONS=3000,SEED=42,FORMAT=pdf,\
OUTPUT_FIG=results/small/similarity_correlation/figure,\
OUTPUT_JSON=results/small/similarity_correlation/results.json,\
OUTPUT_ENV=results/small/similarity_correlation/env.json,\
OUTPUT_REPORT=results/small/similarity_correlation/report.md \
similarity_correlation.sbatch

# medium
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/medium/embeddings.npy,\
SMARTS_FILE=data/embeddings/medium/smarts.txt,\
N_REACTIONS=3000,SEED=42,FORMAT=pdf,\
OUTPUT_FIG=results/medium/similarity_correlation/figure,\
OUTPUT_JSON=results/medium/similarity_correlation/results.json,\
OUTPUT_ENV=results/medium/similarity_correlation/env.json,\
OUTPUT_REPORT=results/medium/similarity_correlation/report.md \
similarity_correlation.sbatch

# large
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/large/embeddings.npy,\
SMARTS_FILE=data/embeddings/large/smarts.txt,\
N_REACTIONS=3000,SEED=42,FORMAT=pdf,\
OUTPUT_FIG=results/large/similarity_correlation/figure,\
OUTPUT_JSON=results/large/similarity_correlation/results.json,\
OUTPUT_ENV=results/large/similarity_correlation/env.json,\
OUTPUT_REPORT=results/large/similarity_correlation/report.md \
similarity_correlation.sbatch
```

---

## Phase 3c-random — Similarity Correlation Baseline (CPU, ~5 min total)

Random-init correlation for all three model sizes in a single job.
Produces `results/final/3c-similarity-correlation/{small,medium,large}_random/results.json`.

The script pre-samples the same 3,000 indices (seed=42, same sort order) as the
pretrained runs so the Tanimoto pairs are identical — only the cosine similarities differ.
Model config JSONs are DVC-tracked; the job pulls them automatically.

```bash
sbatch scripts/similarity_correlation_random_baseline.sbatch
```

> **Pretrained run commands for reference** (already done — outputs in `results/final/3c-similarity-correlation/`):
> ```bash
> # small
> sbatch --export=ALL,\
> EMBEDDINGS=data/embeddings/small/embeddings.npy,\
> SMARTS_FILE=data/embeddings/small/smarts.txt,\
> N_REACTIONS=3000,SEED=42,FORMAT=pdf,\
> OUTPUT_FIG=results/small/similarity_correlation/figure,\
> OUTPUT_JSON=results/small/similarity_correlation/results.json,\
> OUTPUT_ENV=results/small/similarity_correlation/env.json,\
> OUTPUT_REPORT=results/small/similarity_correlation/report.md \
> scripts/similarity_correlation.sbatch
>
> # medium / large — same pattern, change paths accordingly
> ```

---

## Phase 3d — Nearest Neighbors (CPU, <1 min each)

All three can run in parallel. Run from `scripts/`.

```bash
# small
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/small/embeddings.npy,\
SMARTS_FILE=data/embeddings/small/smarts.txt,\
TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42,\
OUTPUT_JSON=results/small/nearest_neighbors/results.json,\
OUTPUT_ENV=results/small/nearest_neighbors/env.json,\
OUTPUT_REPORT=results/small/nearest_neighbors/report.md \
nearest_neighbors.sbatch

# medium
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/medium/embeddings.npy,\
SMARTS_FILE=data/embeddings/medium/smarts.txt,\
TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42,\
OUTPUT_JSON=results/medium/nearest_neighbors/results.json,\
OUTPUT_ENV=results/medium/nearest_neighbors/env.json,\
OUTPUT_REPORT=results/medium/nearest_neighbors/report.md \
nearest_neighbors.sbatch

# large
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/large/embeddings.npy,\
SMARTS_FILE=data/embeddings/large/smarts.txt,\
TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42,\
OUTPUT_JSON=results/large/nearest_neighbors/results.json,\
OUTPUT_ENV=results/large/nearest_neighbors/env.json,\
OUTPUT_REPORT=results/large/nearest_neighbors/report.md \
nearest_neighbors.sbatch
```

---

## Phase 3e — UMAP depth=1 (CPU, ~13 min each — fits UMAP from scratch)

All three can run in parallel. This step also writes the UMAP coordinate cache used in Phase 3f.
Run from `scripts/`.

```bash
# small
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/small/embeddings.npy,\
SMARTS_FILE=data/embeddings/small/smarts.txt,\
EC_DEPTH=1,FORMAT=pdf,\
UMAP_CACHE=results/figures/small_umap_coords.npy,\
OUTPUT=results/small/umap/depth1 \
plot_umap.sbatch

# medium
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/medium/embeddings.npy,\
SMARTS_FILE=data/embeddings/medium/smarts.txt,\
EC_DEPTH=1,FORMAT=pdf,\
UMAP_CACHE=results/figures/medium_umap_coords.npy,\
OUTPUT=results/medium/umap/depth1 \
plot_umap.sbatch

# large
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/large/embeddings.npy,\
SMARTS_FILE=data/embeddings/large/smarts.txt,\
EC_DEPTH=1,FORMAT=pdf,\
UMAP_CACHE=results/figures/large_umap_coords.npy,\
OUTPUT=results/large/umap/depth1 \
plot_umap.sbatch
```

---

## Phase 3f — UMAP depth=2 and depth=3 (CPU, ~30 sec each — reads cached coords)

Wait for the matching Phase 3e job to finish before submitting depths 2 and 3 for each model.
All 6 can then run in parallel. Run from `scripts/`.

```bash
# small depth=2 and depth=3
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/small/embeddings.npy,\
SMARTS_FILE=data/embeddings/small/smarts.txt,\
EC_DEPTH=2,FORMAT=pdf,\
UMAP_CACHE=results/figures/small_umap_coords.npy,\
OUTPUT=results/small/umap/depth2 \
plot_umap.sbatch

sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/small/embeddings.npy,\
SMARTS_FILE=data/embeddings/small/smarts.txt,\
EC_DEPTH=3,FORMAT=pdf,\
UMAP_CACHE=results/figures/small_umap_coords.npy,\
OUTPUT=results/small/umap/depth3 \
plot_umap.sbatch

# medium depth=2 and depth=3
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/medium/embeddings.npy,\
SMARTS_FILE=data/embeddings/medium/smarts.txt,\
EC_DEPTH=2,FORMAT=pdf,\
UMAP_CACHE=results/figures/medium_umap_coords.npy,\
OUTPUT=results/medium/umap/depth2 \
plot_umap.sbatch

sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/medium/embeddings.npy,\
SMARTS_FILE=data/embeddings/medium/smarts.txt,\
EC_DEPTH=3,FORMAT=pdf,\
UMAP_CACHE=results/figures/medium_umap_coords.npy,\
OUTPUT=results/medium/umap/depth3 \
plot_umap.sbatch

# large depth=2 and depth=3
sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/large/embeddings.npy,\
SMARTS_FILE=data/embeddings/large/smarts.txt,\
EC_DEPTH=2,FORMAT=pdf,\
UMAP_CACHE=results/figures/large_umap_coords.npy,\
OUTPUT=results/large/umap/depth2 \
plot_umap.sbatch

sbatch --export=ALL,\
EMBEDDINGS=data/embeddings/large/embeddings.npy,\
SMARTS_FILE=data/embeddings/large/smarts.txt,\
EC_DEPTH=3,FORMAT=pdf,\
UMAP_CACHE=results/figures/large_umap_coords.npy,\
OUTPUT=results/large/umap/depth3 \
plot_umap.sbatch
```

---

## Estimated Wall Time (per model)

| Phase | Job | Time |
|-------|-----|------|
| 1 — Train MLM | 1 GPU job | 3.5 h (small) / 6 h (medium) / ~12–24 h (large) |
| 2 — Extract embeddings | 1 GPU job | ~3 min |
| 3a — Pooling ablation | 1 GPU job | ~47 min |
| 3b — EC classifier (all depths) | 3 CPU jobs in parallel | ~50 min (bottleneck: depth=1) |
| 3c — Similarity correlation | 1 CPU job | ~1 min |
| 3c-random — Random-init baseline | 1 CPU job (all 3 sizes) | ~5 min |
| 3d — Nearest neighbors | 1 CPU job | ~1 min |
| 3e — UMAP depth=1 | 1 CPU job | ~13 min |
| 3f — UMAP depth=2,3 | 2 CPU jobs in parallel | ~30 sec |

Total wall time after training (phases 2–3, all parallel across models): **~50 min**.