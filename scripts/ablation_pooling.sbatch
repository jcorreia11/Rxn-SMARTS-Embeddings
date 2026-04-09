#!/bin/bash
#SBATCH --job-name=ablation_pooling
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=02:00:00
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

# ---- ensure cu124 torch ----
pip install --force-reinstall --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

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

RUN_ID="pooling_ablation_${SLURM_JOB_ID}"
OUTPUT_JSON="${OUTPUT_JSON:-results/${RUN_ID}.json}"
OUTPUT_ENV="${OUTPUT_ENV:-results/${RUN_ID}_env.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-results/${RUN_ID}_report.md}"

mkdir -p results

echo ""
echo "Weights:          ${WEIGHTS}"
echo "Config:           ${CONFIG}"
echo "Vocab:            ${VOCAB}"
echo "N samples:        ${N_SAMPLES}"
echo "EC depth:         ${EC_DEPTH}"
echo "CV folds:         ${FOLDS}"
echo "Embed batch size: ${EMBED_BATCH_SIZE}"
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
echo "=== Job finished on $(date) ==="

deactivate