#!/bin/bash
#SBATCH --job-name=compare_tokenizers
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202508983CPCAA0/jcorreia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import sys; print('sys.path ok, site:', sys.path[0])"
python -c "import sentencepiece; print('sentencepiece:', sentencepiece.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"

# ---- config ----
# Override at submission time if needed:
#   sbatch --export=ALL,N_SAMPLES=20000 compare_tokenizers.sbatch
N_SAMPLES="${N_SAMPLES:-10000}"
EC_DEPTH="${EC_DEPTH:-1}"
SP_MODEL="${SP_MODEL:-data/processed/sp_tokenizer.model}"
SP_VOCAB_SIZE="${SP_VOCAB_SIZE:-1000}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/tokenizer_comparison_${RUN_ID}"
mkdir -p "${RESULTS_DIR}"

echo "Run ID:        ${RUN_ID}"
echo "Results dir:   ${RESULTS_DIR}"
echo "N samples:     ${N_SAMPLES}"
echo "EC depth:      ${EC_DEPTH}"
echo "SP model:      ${SP_MODEL}"
echo "SP vocab size: ${SP_VOCAB_SIZE}"

# ---- step 1: EC classification (trains SP model if missing) ----
echo ""
echo "=== Step 1/3: EC classifier comparison ($(date)) ==="
python scripts/train_ec_classifier.py \
    --n-samples "${N_SAMPLES}" \
    --ec-depth "${EC_DEPTH}" \
    --sp-model "${SP_MODEL}" \
    --sp-vocab-size "${SP_VOCAB_SIZE}" \
    --folds 5 \
    --output "${RESULTS_DIR}/ec_classifier.json"

# ---- step 2: intrinsic metrics ----
echo ""
echo "=== Step 2/3: Intrinsic metrics comparison ($(date)) ==="
python scripts/compare_tokenizers.py \
    --n-samples "${N_SAMPLES}" \
    --sp-model "${SP_MODEL}" \
    --output "${RESULTS_DIR}/tokenizer_metrics.json"

# ---- step 3: generate report ----
echo ""
echo "=== Step 3/3: Generating report ($(date)) ==="
python scripts/generate_report.py \
    --metrics "${RESULTS_DIR}/tokenizer_metrics.json" \
    --classifier "${RESULTS_DIR}/ec_classifier.json" \
    --output "${RESULTS_DIR}/tokenizer_comparison_report.md"

echo ""
echo "Report: ${RESULTS_DIR}/tokenizer_comparison_report.md"
echo "=== Job finished on $(date) ==="

deactivate