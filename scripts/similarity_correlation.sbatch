#!/bin/bash
#SBATCH --job-name=similarity_correlation
#SBATCH --account=f202508983cpcaa0x
#SBATCH --partition=normal-x86
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

# Memory note: 32 GB covers loading the full embedding matrix (~1.4 GB for
# 361k × 256 float32) plus the pairwise cosine matrix for the sampled subset.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202508983CPCAA0/jcorreia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- config ----
# Override at submission time with --export=ALL,VAR=value
EMBEDDINGS="${EMBEDDINGS:-data/embeddings/reaction_embeddings.npy}"
SMARTS_FILE="${SMARTS_FILE:-data/embeddings/reaction_smarts.txt}"
N_REACTIONS="${N_REACTIONS:-1000}"
SEED="${SEED:-42}"
FORMAT="${FORMAT:-pdf}"
DPI="${DPI:-300}"
GRIDSIZE="${GRIDSIZE:-50}"

RUN_ID="sim_corr_${SLURM_JOB_ID}"
OUTPUT_FIG="${OUTPUT_FIG:-results/figures/${RUN_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-results/${RUN_ID}.json}"
OUTPUT_ENV="${OUTPUT_ENV:-results/${RUN_ID}_env.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-results/${RUN_ID}_report.md}"

mkdir -p results/figures

echo ""
echo "Embeddings:  ${EMBEDDINGS}"
echo "SMARTS file: ${SMARTS_FILE}"
echo "N reactions: ${N_REACTIONS}  (~$((N_REACTIONS * (N_REACTIONS - 1) / 2)) pairs)"
echo "Seed:        ${SEED}"
echo "Figure:      ${OUTPUT_FIG}.${FORMAT}"
echo "JSON:        ${OUTPUT_JSON}"
echo "Report:      ${OUTPUT_REPORT}"

# ---- collect environment info ----
echo ""
echo "=== Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${OUTPUT_ENV}"

# ---- run similarity analysis ----
echo ""
echo "=== Running similarity correlation analysis ($(date)) ==="
python scripts/similarity_correlation.py \
    --embeddings  "${EMBEDDINGS}" \
    --smarts      "${SMARTS_FILE}" \
    --n-reactions "${N_REACTIONS}" \
    --seed        "${SEED}" \
    --output      "${OUTPUT_FIG}" \
    --format      "${FORMAT}" \
    --dpi         "${DPI}" \
    --gridsize    "${GRIDSIZE}" \
    --save-json   "${OUTPUT_JSON}"

# ---- generate report ----
echo ""
echo "=== Generating report ($(date)) ==="
python scripts/generate_similarity_report.py \
    --results "${OUTPUT_JSON}" \
    --env     "${OUTPUT_ENV}" \
    --figure  "${OUTPUT_FIG}.${FORMAT}" \
    --output  "${OUTPUT_REPORT}"

echo ""
echo "Figure:      ${OUTPUT_FIG}.${FORMAT}"
echo "Stats:       ${OUTPUT_JSON}"
echo "Environment: ${OUTPUT_ENV}"
echo "Report:      ${OUTPUT_REPORT}"

# ---- commit results to git ----
echo ""
echo "=== Committing results to git ($(date)) ==="
git add "${OUTPUT_FIG}.${FORMAT}" "${OUTPUT_JSON}" "${OUTPUT_ENV}" "${OUTPUT_REPORT}"
git commit -m "[results] Similarity correlation (run ${RUN_ID}, N=${N_REACTIONS})"

echo "=== Job finished on $(date) ==="

deactivate