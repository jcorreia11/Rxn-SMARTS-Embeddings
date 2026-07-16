#!/bin/bash
#SBATCH --job-name=ec_classifier
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202508983CPCAA0/jcorreia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

# ---- debug: environment info ----
echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi not available"
echo "Python executable: $(which python)"
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---- config ----
# Override at submission time with --export=ALL,VAR=value
VALIDATED="${VALIDATED:-data/processed/validated_smarts.csv}"
SP_MODEL="${SP_MODEL:-data/processed/sp_tokenizer.model}"
EC_DEPTH="${EC_DEPTH:-1}"
N_SAMPLES="${N_SAMPLES:-10000}"
FOLDS="${FOLDS:-5}"
# 'group' (default) keeps RetroRules radius-siblings together, avoiding
# train/test leakage. 'random' reproduces the old leaky split — only for
# before/after comparison via generate_split_comparison_report.py.
SPLIT_STRATEGY="${SPLIT_STRATEGY:-group}"
# Label only (small/medium/large) — used with EC_DEPTH to mirror artifacts
# into results/final/3b-ec-classifier/${MODEL_SIZE}/depth${EC_DEPTH}/.
MODEL_SIZE="${MODEL_SIZE:-unspecified}"

# Embedding experiment — set these to enable pretrained/random embedding runs.
# Leave WEIGHTS empty to skip embedding experiments and run TF-IDF only.
WEIGHTS="${WEIGHTS:-}"
CONFIG="${CONFIG:-}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
POOLING="${POOLING:-mean}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/ec_classifier_${RUN_ID}.json"
ENV_JSON="results/ec_classifier_${RUN_ID}_env.json"
REPORT="results/ec_classifier_${RUN_ID}_report.md"
mkdir -p results

echo ""
echo "Run ID:           ${RUN_ID}"
echo "Validated SMARTS: ${VALIDATED}"
echo "SP model:         ${SP_MODEL}"
echo "EC depth:         ${EC_DEPTH}"
echo "N samples:        ${N_SAMPLES}"
echo "Folds:            ${FOLDS}"
echo "Split strategy:   ${SPLIT_STRATEGY}"
echo "Model size:       ${MODEL_SIZE}"
echo "Results JSON:     ${OUTPUT}"
echo "Environment JSON: ${ENV_JSON}"
echo "Report:           ${REPORT}"
if [[ -n "${WEIGHTS}" ]]; then
    echo "Weights:          ${WEIGHTS}"
    echo "Config:           ${CONFIG}"
    echo "Vocab:            ${VOCAB}"
    echo "Pooling:          ${POOLING}"
    echo "Embed batch size: ${EMBED_BATCH_SIZE}"
else
    echo "Embedding experiments: SKIPPED (set WEIGHTS to enable)"
fi

# ---- collect environment on the compute node (has GPU) ----
echo ""
echo "=== Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${ENV_JSON}"

# ---- build base args ----
ARGS=(
    --validated       "${VALIDATED}"
    --sp-model        "${SP_MODEL}"
    --ec-depth        "${EC_DEPTH}"
    --n-samples       "${N_SAMPLES}"
    --folds           "${FOLDS}"
    --split-strategy  "${SPLIT_STRATEGY}"
    --output          "${OUTPUT}"
)

# ---- append embedding args if WEIGHTS is set ----
if [[ -n "${WEIGHTS}" ]]; then
    ARGS+=(
        --weights          "${WEIGHTS}"
        --config           "${CONFIG}"
        --vocab            "${VOCAB}"
        --pooling          "${POOLING}"
        --embed-batch-size "${EMBED_BATCH_SIZE}"
    )
fi

# ---- run ----
echo ""
echo "=== Running EC classifier ($(date)) ==="
python scripts/train_ec_classifier.py "${ARGS[@]}"

# ---- generate report ----
echo ""
echo "=== Generating report ($(date)) ==="
python scripts/generate_ec_report.py \
    --results       "${OUTPUT}" \
    --env           "${ENV_JSON}" \
    --output        "${REPORT}" \
    --figure-format pdf

echo ""
echo "Results JSON:  ${OUTPUT}"
echo "Environment:   ${ENV_JSON}"
echo "Report:        ${REPORT}"
echo "Figure:        ${REPORT%.md}.pdf"

# ---- mirror into results/final/ for easy browsing ----
# (results/ is gitignored — neither the flat files above nor this mirror
# are committed to git. Both are local-only; back them up via rsync/scp.)
echo ""
echo "=== Mirroring artifacts into results/final/3b-ec-classifier/${MODEL_SIZE}/depth${EC_DEPTH}/ ($(date)) ==="
MIRROR_DIR="results/final/3b-ec-classifier/${MODEL_SIZE}/depth${EC_DEPTH}"
mkdir -p "${MIRROR_DIR}"
cp "${OUTPUT}" "${MIRROR_DIR}/results.json"
cp "${ENV_JSON}" "${MIRROR_DIR}/env.json"
cp "${REPORT}" "${MIRROR_DIR}/report.md"
cp "${REPORT%.md}.pdf" "${MIRROR_DIR}/report.pdf"
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate