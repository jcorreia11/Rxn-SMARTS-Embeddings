#!/bin/bash
#SBATCH --job-name=plot_dataset_stats
#SBATCH --account=f202508983cpcaa0x
#SBATCH --partition=normal-x86
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

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
# Override at submission time if needed:
#   sbatch --export=ALL,VAL_SPLIT=0.1 plot_dataset_stats.sbatch
VALIDATED="${VALIDATED:-data/processed/validated_smarts.csv}"
RAW_FILES="${RAW_FILES:-data/raw/retrorules-v3.0-rhea.csv data/raw/retrorules-v3.0-metanetx.csv}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
DPI="${DPI:-300}"
OUTPUT="${OUTPUT:-results/figures/dataset_overview}"
# Only used by plot_length_distributions.py below (Phase 0 outputs).
VOCAB="${VOCAB:-data/processed/vocab.json}"
SP_MODEL="${SP_MODEL:-data/processed/sp_tokenizer.model}"
LENGTH_DIST_OUTPUT="${LENGTH_DIST_OUTPUT:-results/figures/length_distributions}"

echo ""
echo "Validated SMARTS:    ${VALIDATED}"
echo "Raw files:           ${RAW_FILES}"
echo "Val split:           ${VAL_SPLIT}"
echo "DPI:                 ${DPI}"
echo "Output stem:         ${OUTPUT}"
echo "Vocab:               ${VOCAB}"
echo "SP model:            ${SP_MODEL}"
echo "Length dist. output: ${LENGTH_DIST_OUTPUT}"

mkdir -p "$(dirname "${OUTPUT}")"

# ---- generate figures ----
# Saves: panel_a/b/c as SVG + PNG, combined as PDF
echo ""
echo "=== Generating dataset overview figures ($(date)) ==="
python scripts/plot_dataset_stats.py \
    --validated "${VALIDATED}" \
    --raw       ${RAW_FILES} \
    --val-split "${VAL_SPLIT}" \
    --output    "${OUTPUT}" \
    --dpi       "${DPI}"

FIGURES_DIR="$(dirname "${OUTPUT}")"
echo ""
echo "Figures saved to: ${FIGURES_DIR}/"

# ---- generate sequence length distribution figures (Methods 2.1) ----
# Depends only on Phase 0 outputs — no trained model needed.
echo ""
echo "=== Generating sequence length distribution figures ($(date)) ==="
python paper/plot_length_distributions.py \
    --validated "${VALIDATED}" \
    --vocab     "${VOCAB}" \
    --sp-model  "${SP_MODEL}" \
    --output    "${LENGTH_DIST_OUTPUT}" \
    --dpi       "${DPI}"

# ---- generate tokenizer example figure (Methods 2.2.1) ----
# No data/model dependency at all (fixed example SMARTS + rule-based
# tokenizer only) — writes paper/figures/tokenizer_example.{pdf,png},
# mirrored into results/final/4-paper/ by Phase 6's sync_to_results_final.py.
echo ""
echo "=== Generating tokenizer example figure ($(date)) ==="
python paper/plot_tokenizer_example.py

# ---- commit results to git ----
echo ""
echo "=== Committing results to git ($(date)) ==="
git add "${FIGURES_DIR}/panel_a_smarts_length."* \
        "${FIGURES_DIR}/panel_b_ec_distribution."* \
        "${FIGURES_DIR}/panel_c_train_val_split."* \
        "${OUTPUT}.pdf" \
        "${FIGURES_DIR}/length_dist_char."* \
        "${FIGURES_DIR}/length_dist_rule_tok."* \
        "${FIGURES_DIR}/length_dist_sp_tok."* \
        "${LENGTH_DIST_OUTPUT}.pdf" \
        "${LENGTH_DIST_OUTPUT}.json" \
        "paper/figures/tokenizer_example.pdf" \
        "paper/figures/tokenizer_example.png"
git commit -m "[results] Dataset overview, length distribution, and tokenizer example figures (job ${SLURM_JOB_ID})"

# ---- mirror into results/final/ for easy browsing ----
# (local convenience copy only, gitignored — the flat results/figures/ and
# paper/figures/ files above, git-tracked, remain the actual backed-up record)
echo ""
echo "=== Mirroring artifacts into results/final/0-dataset-stats/ ($(date)) ==="
MIRROR_DIR="results/final/0-dataset-stats"
mkdir -p "${MIRROR_DIR}"
cp "${FIGURES_DIR}/panel_a_smarts_length."* \
   "${FIGURES_DIR}/panel_b_ec_distribution."* \
   "${FIGURES_DIR}/panel_c_train_val_split."* \
   "${OUTPUT}.pdf" \
   "${FIGURES_DIR}/length_dist_char."* \
   "${FIGURES_DIR}/length_dist_rule_tok."* \
   "${FIGURES_DIR}/length_dist_sp_tok."* \
   "${LENGTH_DIST_OUTPUT}.pdf" \
   "${LENGTH_DIST_OUTPUT}.json" \
   "paper/figures/tokenizer_example.pdf" \
   "paper/figures/tokenizer_example.png" \
   "${MIRROR_DIR}/"
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate