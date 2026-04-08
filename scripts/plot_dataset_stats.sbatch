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
#   sbatch --export=ALL,VAL_SPLIT=0.1,FORMAT=png plot_dataset_stats.sbatch
VALIDATED="${VALIDATED:-data/processed/validated_smarts.csv}"
RAW_FILES="${RAW_FILES:-data/raw/retrorules-v3.0-rhea.csv data/raw/retrorules-v3.0-metanetx.csv}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
FORMAT="${FORMAT:-pdf}"
DPI="${DPI:-300}"
OUTPUT="${OUTPUT:-results/figures/dataset_overview}"

echo ""
echo "Validated SMARTS: ${VALIDATED}"
echo "Raw files:        ${RAW_FILES}"
echo "Val split:        ${VAL_SPLIT}"
echo "Format:           ${FORMAT}"
echo "Output:           ${OUTPUT}.${FORMAT}"

mkdir -p "$(dirname "${OUTPUT}")"

# ---- generate figure ----
echo ""
echo "=== Generating dataset overview figure ($(date)) ==="
python scripts/plot_dataset_stats.py \
    --validated "${VALIDATED}" \
    --raw       ${RAW_FILES} \
    --val-split "${VAL_SPLIT}" \
    --output    "${OUTPUT}" \
    --format    "${FORMAT}" \
    --dpi       "${DPI}"

echo ""
echo "Figure: ${OUTPUT}.${FORMAT}"
echo "=== Job finished on $(date) ==="

deactivate