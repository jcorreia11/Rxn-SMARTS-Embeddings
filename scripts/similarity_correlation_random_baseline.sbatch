#!/bin/bash
#SBATCH --job-name=sim_corr_random
#SBATCH --account=f202508983cpcaa0x
#SBATCH --partition=normal-x86
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

# Runs the similarity–structural correlation analysis for random-initialised
# (untrained) models at all three sizes (small / medium / large) using the
# same 3,000-reaction sample and seed=42 as the pretrained runs, so the
# Tanimoto values and pair identity are identical.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202508983CPCAA0/jcorreia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- pull model config JSONs (DVC-tracked, small files) ----
echo ""
echo "=== Pulling model config JSONs via DVC ($(date)) ==="
dvc pull \
    models/smarts_transformer_20260410_111830.json \
    models/smarts_transformer_20260428_160310.json \
    models/smarts_transformer_20260513_093030.json

# ---- shared config ----
N_REACTIONS="${N_REACTIONS:-3000}"
SEED="${SEED:-42}"
MODEL_SEED="${MODEL_SEED:-42}"
FORMAT="${FORMAT:-pdf}"
DPI="${DPI:-300}"
GRIDSIZE="${GRIDSIZE:-50}"

VOCAB="data/processed/vocab.json"
OUT_BASE="results/final/3c-similarity-correlation"
SUFFIX="${SUFFIX:-_random_s${MODEL_SEED}}"

RUN_ID="sim_corr_random_${SLURM_JOB_ID}"

# ============================================================
# small  (6L, d=256, max_len=256)
# ============================================================
echo ""
echo "=== small model (model-seed=${MODEL_SEED}, $(date)) ==="
python scripts/similarity_correlation_random_baseline.py \
    --config      "models/smarts_transformer_20260410_111830.json" \
    --vocab       "${VOCAB}" \
    --smarts      "data/embeddings/small/smarts.txt" \
    --n-reactions "${N_REACTIONS}" \
    --seed        "${SEED}" \
    --model-seed  "${MODEL_SEED}" \
    --output      "${OUT_BASE}/small${SUFFIX}/figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${OUT_BASE}/small${SUFFIX}/results.json"

# ============================================================
# medium  (6L, d=256, max_len=512)
# ============================================================
echo ""
echo "=== medium model (model-seed=${MODEL_SEED}, $(date)) ==="
python scripts/similarity_correlation_random_baseline.py \
    --config      "models/smarts_transformer_20260428_160310.json" \
    --vocab       "${VOCAB}" \
    --smarts      "data/embeddings/medium/smarts.txt" \
    --n-reactions "${N_REACTIONS}" \
    --seed        "${SEED}" \
    --model-seed  "${MODEL_SEED}" \
    --output      "${OUT_BASE}/medium${SUFFIX}/figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${OUT_BASE}/medium${SUFFIX}/results.json"

# ============================================================
# large  (8L, d=512, max_len=512)
# ============================================================
echo ""
echo "=== large model (model-seed=${MODEL_SEED}, $(date)) ==="
python scripts/similarity_correlation_random_baseline.py \
    --config      "models/smarts_transformer_20260513_093030.json" \
    --vocab       "${VOCAB}" \
    --smarts      "data/embeddings/large/smarts.txt" \
    --n-reactions "${N_REACTIONS}" \
    --seed        "${SEED}" \
    --model-seed  "${MODEL_SEED}" \
    --output      "${OUT_BASE}/large${SUFFIX}/figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${OUT_BASE}/large${SUFFIX}/results.json"

# ---- commit results to git ----
echo ""
echo "=== Committing results ($(date)) ==="
git add \
    "${OUT_BASE}/small${SUFFIX}/"  \
    "${OUT_BASE}/medium${SUFFIX}/" \
    "${OUT_BASE}/large${SUFFIX}/"
git commit -m "[results] Similarity correlation — random-init baseline (${RUN_ID}, model-seed=${MODEL_SEED}, N=${N_REACTIONS})"

echo "=== Job finished on $(date) ==="

deactivate