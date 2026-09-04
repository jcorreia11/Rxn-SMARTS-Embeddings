#!/bin/bash
#SBATCH --job-name=compare_tokenizers
#SBATCH --account=f202607628cpcaa1x
#SBATCH --partition=normal-x86
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/SmartRxnEmbeddings

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
# Deliberately 5, not the 10 used by the main train_ec_classifier phase (3b) —
# this is a lighter-weight tokenizer probe, not the headline EC result; see
# hpc-runbook.md's "publication-quality argument choices" table.
FOLDS="${FOLDS:-5}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/tokenizer_comparison_${RUN_ID}"
FIGURES_DIR="${RESULTS_DIR}/figures"
ENV_JSON="${RESULTS_DIR}/env.json"
mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

echo "Run ID:        ${RUN_ID}"
echo "Results dir:   ${RESULTS_DIR}"
echo "N samples:     ${N_SAMPLES}"
echo "EC depth:      ${EC_DEPTH}"
echo "SP model:      ${SP_MODEL}"
echo "SP vocab size: ${SP_VOCAB_SIZE}"
echo "CV folds:      ${FOLDS}"

# ---- step 1: collect environment info ----
echo ""
echo "=== Step 1/5: Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${ENV_JSON}"

# ---- step 2: EC classification (trains SP model if missing) ----
echo ""
echo "=== Step 2/5: EC classifier comparison ($(date)) ==="
python scripts/train_ec_classifier.py \
    --n-samples "${N_SAMPLES}" \
    --ec-depth "${EC_DEPTH}" \
    --sp-model "${SP_MODEL}" \
    --sp-vocab-size "${SP_VOCAB_SIZE}" \
    --folds "${FOLDS}" \
    --output "${RESULTS_DIR}/ec_classifier.json"

# ---- step 3: intrinsic metrics ----
echo ""
echo "=== Step 3/5: Intrinsic metrics comparison ($(date)) ==="
python scripts/compare_tokenizers.py \
    --n-samples "${N_SAMPLES}" \
    --sp-model "${SP_MODEL}" \
    --output "${RESULTS_DIR}/tokenizer_metrics.json"

# ---- step 4: generate figures ----
echo ""
echo "=== Step 4/5: Generating figures ($(date)) ==="
python scripts/plot_tokenizer_comparison.py \
    --metrics    "${RESULTS_DIR}/tokenizer_metrics.json" \
    --classifier "${RESULTS_DIR}/ec_classifier.json" \
    --output-dir "${FIGURES_DIR}" \
    --dpi        300

# ---- step 5: generate report ----
echo ""
echo "=== Step 5/5: Generating report ($(date)) ==="
python scripts/generate_report.py \
    --metrics     "${RESULTS_DIR}/tokenizer_metrics.json" \
    --classifier  "${RESULTS_DIR}/ec_classifier.json" \
    --env         "${ENV_JSON}" \
    --figures-dir "${FIGURES_DIR}" \
    --output      "${RESULTS_DIR}/tokenizer_comparison_report.md"

echo ""
echo "Figures:  ${FIGURES_DIR}/"
echo "Report:   ${RESULTS_DIR}/tokenizer_comparison_report.md"

# ---- mirror into results/final/ for easy browsing ----
# (results/ is gitignored — neither ${RESULTS_DIR} above nor this mirror
# are committed to git. Both are local-only; back them up via rsync/scp.
# Stable "tokenizer_comparison" name, not RUN_ID-based, so
# paper/dataset_tokenizer_results.py's own latest-dir lookup and this
# mirror don't need to agree on which RUN_ID is current.)
echo ""
echo "=== Mirroring artifacts into results/final/0-dataset-stats/tokenizer_comparison/ ($(date)) ==="
MIRROR_DIR="results/final/0-dataset-stats/tokenizer_comparison"
mkdir -p "${MIRROR_DIR}"
cp -r "${RESULTS_DIR}/." "${MIRROR_DIR}/"
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate