# Project Status and Recommended Experiments

## What's Already Done
- Phase 0–4 complete
- EC depth=1: 83.8% (Pretrained+MLP)
- EC depth=2: 74.9% (Pretrained+MLP)
- Similarity: Pearson r = 0.61 (4.5M pairs)
- UMAP + nearest neighbors done

---

## Recommended Experiments

### 1. Retrain MAX_LENGTH=512 (CRITICAL, ~4h GPU)
Current 256 covers ~75%; 512 covers ~98% → required for publication.

Then re-run:
- extract_embeddings (~30m GPU)
- plot_umap, similarity_correlation, train_ec_classifier (CPU)

Command:
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=512,D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4 scripts/train_mlm.sbatch

---

### 2. EC depth=2 classifier (~25m CPU, ready)
Command:
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,POOLING=mean,N_SAMPLES=50000,EC_DEPTH=2,FOLDS=10,EMBED_BATCH_SIZE=256 scripts/train_ec_classifier.sbatch

---

### 3. EC depth=3 classifier (~35m CPU)
Shows degradation across label granularity.

Command:
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,POOLING=mean,N_SAMPLES=50000,EC_DEPTH=3,FOLDS=10,EMBED_BATCH_SIZE=256 scripts/train_ec_classifier.sbatch

---

### 4. Larger model (d_model=512, ~8–10h GPU)
Scaling result vs current (~5M params).

Settings:
- d_model=512, nhead=16, layers=8, ffn=2048, MAX_LENGTH=512

Command:
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=32,LR=1e-4,MAX_LENGTH=512,D_MODEL=512,NHEAD=16,NUM_LAYERS=8,DIM_FEEDFORWARD=2048,DROPOUT=0.1,MASK_PROB=0.25,MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4 scripts/train_mlm.sbatch

Note: reduce BATCH_SIZE to 16 if OOM (A100 40GB should handle 32).

---

### 5. UMAP with EC depth=2 (~10m CPU)
Re-run plot_umap with EC_DEPTH=2 (small script tweak needed).

---

## Summary

| # | Experiment | Compute | Effort | Value |
|---|-----------|--------|--------|-------|
| 1 | MAX_LENGTH=512 retrain | ~5–6h GPU + CPU | low | CRITICAL |
| 2 | EC depth=2 | ~25m CPU | trivial | easy win |
| 3 | EC depth=3 | ~35m CPU | trivial | good paper |
| 4 | Larger model | ~8–10h GPU | low | strong ablation |
| 5 | UMAP depth=2 | ~10m CPU | minor edit | nice figure |

---

## Final Notes
- Run 2 & 3 now (no changes needed)
- 1 is mandatory for correctness
- 4 is best use of remaining GPU budget (~10h)

sbatch --export=ALL,EMBEDDINGS=data/embeddings/reaction_embeddings.npy,SMARTS_FILE=data/embeddings/reaction_smarts.txt,EC_DEPTH=2,FORMAT=pdf,DPI=300 plot_umap.sbatch 