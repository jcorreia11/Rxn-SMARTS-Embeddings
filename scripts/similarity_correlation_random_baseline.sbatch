#!/bin/bash
#SBATCH --job-name=sim_corr_random
#SBATCH --account=f202607628cpcaa1x
#SBATCH --partition=normal-x86
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

# Runs the similarity–structural correlation analysis for random-initialised
# (untrained) models at all three sizes (small / medium / large) using the
# same 3,000-reaction sample and seed=42 as the pretrained runs, so the
# Tanimoto values and pair identity are identical.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/Rxn-SMARTS-Embeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- shared config ----
N_REACTIONS="${N_REACTIONS:-3000}"
SEED="${SEED:-42}"
MODEL_SEED="${MODEL_SEED:-42}"
FORMAT="${FORMAT:-pdf}"
DPI="${DPI:-300}"
GRIDSIZE="${GRIDSIZE:-50}"

# Only the architecture (model_config) is read from these — random weights
# are generated fresh from MODEL_SEED, no checkpoint is loaded. Still, these
# configs are gitignored, local-only artifacts (see extract_embeddings.sbatch)
# tied to a specific Phase 1 RUN_ID, so they do NOT auto-update when Phase 1
# is re-run. After retraining, override with e.g.
#   --export=ALL,SMALL_CONFIG=${SMALL_CONFIG},MEDIUM_CONFIG=${MEDIUM_CONFIG},LARGE_CONFIG=${LARGE_CONFIG}
# (see hpc-runbook.md Phase 4b) — the defaults below are just the last known-good run IDs.
SMALL_CONFIG="${SMALL_CONFIG:-models/smarts_transformer_20260715_110917.json}"
MEDIUM_CONFIG="${MEDIUM_CONFIG:-models/smarts_transformer_20260715_111009.json}"
LARGE_CONFIG="${LARGE_CONFIG:-models/smarts_transformer_20260715_111042.json}"

VOCAB="data/processed/vocab.json"
SUFFIX="${SUFFIX:-_random_s${MODEL_SEED}}"

echo ""
echo "Small config:  ${SMALL_CONFIG}"
echo "Medium config: ${MEDIUM_CONFIG}"
echo "Large config:  ${LARGE_CONFIG}"

RUN_ID="sim_corr_random_${SLURM_JOB_ID}"

# Flat outputs, mirrored into OUT_BASE below for easy browsing. Both are
# gitignored (results/*), local-only artifacts — back them up via rsync/scp.
FLAT_BASE="results/${RUN_ID}"
OUT_BASE="results/final/3c-similarity-correlation"
mkdir -p results

# ============================================================
# small  (6L, d=256, max_len=256)
# ============================================================
echo ""
echo "=== small model (model-seed=${MODEL_SEED}, $(date)) ==="
python scripts/similarity_correlation_random_baseline.py \
    --config      "${SMALL_CONFIG}" \
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
    --config      "${MEDIUM_CONFIG}" \
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
    --config      "${LARGE_CONFIG}" \
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

# ---- mirror into results/final/ for easy browsing ----
# (results/ is gitignored — neither the flat files above nor this mirror
# are committed to git. Both are local-only; back them up via rsync/scp.)
echo ""
echo "=== Mirroring artifacts into ${OUT_BASE}/{small,medium,large}${SUFFIX}/ ($(date)) ==="
for size in small medium large; do
    MIRROR_DIR="${OUT_BASE}/${size}${SUFFIX}"
    mkdir -p "${MIRROR_DIR}"
    cp "${FLAT_BASE}_${size}${SUFFIX}.json" "${MIRROR_DIR}/results.json"
    cp "${FLAT_BASE}_${size}${SUFFIX}_figure.${FORMAT}" "${MIRROR_DIR}/figure.${FORMAT}"
    cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
    cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true
done

echo "=== Job finished on $(date) ==="

deactivate