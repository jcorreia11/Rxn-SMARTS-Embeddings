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
SUFFIX="${SUFFIX:-_random_s${MODEL_SEED}}"

RUN_ID="sim_corr_random_${SLURM_JOB_ID}"

# Flat, git-tracked primary outputs (the actual backed-up record).
# OUT_BASE below is a gitignored local mirror only — a prior version of
# this script wrote *only* to OUT_BASE, so its "commit results" step was a
# silent no-op (nothing to add) and, under `set -euo pipefail`, `git commit`
# on an empty index exits non-zero and fails the whole job even though the
# analysis itself succeeded. Writing the flat copies first fixes that.
FLAT_BASE="results/${RUN_ID}"
OUT_BASE="results/final/3c-similarity-correlation"
mkdir -p results

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
    --output      "${FLAT_BASE}_small${SUFFIX}_figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${FLAT_BASE}_small${SUFFIX}.json"

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
    --output      "${FLAT_BASE}_medium${SUFFIX}_figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${FLAT_BASE}_medium${SUFFIX}.json"

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
    --output      "${FLAT_BASE}_large${SUFFIX}_figure" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${FLAT_BASE}_large${SUFFIX}.json"

# ---- commit results to git ----
echo ""
echo "=== Committing results ($(date)) ==="
git add \
    "${FLAT_BASE}_small${SUFFIX}.json"  "${FLAT_BASE}_small${SUFFIX}_figure.${FORMAT}" \
    "${FLAT_BASE}_medium${SUFFIX}.json" "${FLAT_BASE}_medium${SUFFIX}_figure.${FORMAT}" \
    "${FLAT_BASE}_large${SUFFIX}.json"  "${FLAT_BASE}_large${SUFFIX}_figure.${FORMAT}"
git commit -m "[results] Similarity correlation — random-init baseline (${RUN_ID}, model-seed=${MODEL_SEED}, N=${N_REACTIONS})"

# ---- mirror into results/final/ for easy browsing ----
# (local convenience copy only, gitignored — the flat results/ files above,
# git-tracked, remain the actual backed-up record)
echo ""
echo "=== Mirroring artifacts into ${OUT_BASE}/{small,medium,large}${SUFFIX}/ ($(date)) ==="
for size in small medium large; do
    MIRROR_DIR="${OUT_BASE}/${size}${SUFFIX}"
    mkdir -p "${MIRROR_DIR}"
    cp "${FLAT_BASE}_${size}${SUFFIX}.json" "${MIRROR_DIR}/results.json"
    cp "${FLAT_BASE}_${size}${SUFFIX}_figure.${FORMAT}" "${MIRROR_DIR}/figure.${FORMAT}"
    cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
    cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true
done

echo "=== Job finished on $(date) ==="

deactivate