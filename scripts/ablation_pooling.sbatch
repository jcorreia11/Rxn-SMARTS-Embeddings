#!/bin/bash
#SBATCH --job-name=ablation_pooling
#SBATCH --account=f202607628cpcaa1g
#SBATCH --partition=normal-a100-40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

# ---- ensure cu124 torch ----
TORCH_BUILD=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ "$TORCH_BUILD" != *"cu124"* ]]; then
    echo "=== Reinstalling torch for cu124 ==="
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "=== torch cu124 already present — skipping reinstall ==="
fi

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---- config ----
# Override at submission time with --export=ALL,VAR=value
WEIGHTS="${WEIGHTS:-models/smarts_transformer.pt}"
CONFIG="${CONFIG:-${WEIGHTS%.pt}.json}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
N_SAMPLES="${N_SAMPLES:-10000}"
EC_DEPTH="${EC_DEPTH:-1}"
FOLDS="${FOLDS:-5}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
# 'group' (default) keeps RetroRules radius-siblings together, avoiding
# train/test leakage. 'random' reproduces the old leaky split — only for
# before/after comparison via generate_split_comparison_report.py.
SPLIT_STRATEGY="${SPLIT_STRATEGY:-group}"
# Label only (small/medium/large) — used to mirror artifacts into
# results/final/3a-pooling-ablation/${MODEL_SIZE}/ for easy browsing.
MODEL_SIZE="${MODEL_SIZE:-unspecified}"

RUN_ID="pooling_ablation_${SLURM_JOB_ID}"
OUTPUT_JSON="${OUTPUT_JSON:-results/${RUN_ID}.json}"
OUTPUT_ENV="${OUTPUT_ENV:-results/${RUN_ID}_env.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-results/${RUN_ID}_report.md}"
OUTPUT_FIGURE="${OUTPUT_REPORT%.md}.pdf"

mkdir -p results

echo ""
echo "Weights:          ${WEIGHTS}"
echo "Config:           ${CONFIG}"
echo "Vocab:            ${VOCAB}"
echo "N samples:        ${N_SAMPLES}"
echo "EC depth:         ${EC_DEPTH}"
echo "CV folds:         ${FOLDS}"
echo "Embed batch size: ${EMBED_BATCH_SIZE}"
echo "Split strategy:   ${SPLIT_STRATEGY}"
echo "Model size:       ${MODEL_SIZE}"
echo "Results JSON:     ${OUTPUT_JSON}"
echo "Report:           ${OUTPUT_REPORT}"

# ---- collect environment info ----
echo ""
echo "=== Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${OUTPUT_ENV}"

# ---- run ablation ----
echo ""
echo "=== Running CLS vs mean pooling ablation ($(date)) ==="
python scripts/ablation_pooling.py \
    --weights           "${WEIGHTS}" \
    --config            "${CONFIG}" \
    --vocab             "${VOCAB}" \
    --n-samples         "${N_SAMPLES}" \
    --ec-depth          "${EC_DEPTH}" \
    --folds             "${FOLDS}" \
    --embed-batch-size  "${EMBED_BATCH_SIZE}" \
    --split-strategy    "${SPLIT_STRATEGY}" \
    --output            "${OUTPUT_JSON}"

# ---- generate report ----
echo ""
echo "=== Generating report ($(date)) ==="
python scripts/generate_pooling_report.py \
    --results "${OUTPUT_JSON}" \
    --env     "${OUTPUT_ENV}" \
    --output  "${OUTPUT_REPORT}"

echo ""
echo "Results JSON: ${OUTPUT_JSON}"
echo "Environment:  ${OUTPUT_ENV}"
echo "Report:       ${OUTPUT_REPORT}"

# ---- mirror into results/final/ for easy browsing ----
# (results/ is gitignored — neither the flat files above nor this mirror
# are committed to git. Both are local-only; back them up via rsync/scp.)
echo ""
echo "=== Mirroring artifacts into results/final/3a-pooling-ablation/${MODEL_SIZE}/ ($(date)) ==="
MIRROR_DIR="results/final/3a-pooling-ablation/${MODEL_SIZE}"
mkdir -p "${MIRROR_DIR}"
cp "${OUTPUT_JSON}" "${MIRROR_DIR}/results.json"
cp "${OUTPUT_ENV}" "${MIRROR_DIR}/env.json"
cp "${OUTPUT_REPORT}" "${MIRROR_DIR}/report.md"
cp "${OUTPUT_FIGURE}" "${MIRROR_DIR}/report.pdf"
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate