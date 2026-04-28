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

echo ""
echo "Validated SMARTS: ${VALIDATED}"
echo "Raw files:        ${RAW_FILES}"
echo "Val split:        ${VAL_SPLIT}"
echo "DPI:              ${DPI}"
echo "Output stem:      ${OUTPUT}"

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

# ---- commit results to git ----
echo ""
echo "=== Committing results to git ($(date)) ==="
git add "${FIGURES_DIR}/panel_a_smarts_length."* \
        "${FIGURES_DIR}/panel_b_ec_distribution."* \
        "${FIGURES_DIR}/panel_c_train_val_split."* \
        "${OUTPUT}.pdf"
git commit -m "[results] Dataset overview figures — SVG+PNG panels + combined PDF (job ${SLURM_JOB_ID})"

echo "=== Job finished on $(date) ==="

deactivate