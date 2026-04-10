#!/bin/bash
#SBATCH --job-name=train_mlm
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=12:00:00
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
echo "=== pip config ==="
pip config list
echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi not available"
echo "=== torch check ==="
TORCH_BUILD=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
echo "Installed torch: ${TORCH_BUILD:-not found}"
if [[ "$TORCH_BUILD" != *"cu124"* ]]; then
    echo "=== Reinstalling torch for cu124 ==="
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo "=== torch after install ==="
    python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
else
    echo "=== torch cu124 already present — skipping reinstall ==="
fi

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---- config ----
# Override at submission time if needed:
#   sbatch --export=ALL,EPOCHS=50,D_MODEL=256 train_mlm.sbatch
DATA="${DATA:-data/processed/validated_smarts.csv}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
MAX_LENGTH="${MAX_LENGTH:-256}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-40}"
MASK_PROB="${MASK_PROB:-0.25}"
MEAN_SPAN="${MEAN_SPAN:-3.0}"
MAX_SPAN="${MAX_SPAN:-10}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="models/checkpoints/${RUN_ID}"
OUTPUT="models/smarts_transformer_${RUN_ID}.pt"
mkdir -p "${CHECKPOINT_DIR}"

echo ""
echo "Run ID:          ${RUN_ID}"
echo "Data:            ${DATA}"
echo "Vocab:           ${VOCAB}"
echo "Max length:      ${MAX_LENGTH}"
echo "d_model:         ${D_MODEL}"
echo "nhead:           ${NHEAD}"
echo "Layers:          ${NUM_LAYERS}"
echo "FFN dim:         ${DIM_FEEDFORWARD}"
echo "Dropout:         ${DROPOUT}"
echo "Learning rate:   ${LR}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Epochs:          ${EPOCHS}"
echo "Mask prob:       ${MASK_PROB}"
echo "Mean span:       ${MEAN_SPAN}"
echo "Max span:        ${MAX_SPAN}"
echo "Val split:       ${VAL_SPLIT}"
echo "Warmup steps:    ${WARMUP_STEPS}"
echo "Max grad norm:   ${MAX_GRAD_NORM}"
echo "Num workers:     ${NUM_WORKERS}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Output:          ${OUTPUT}"

# ---- train ----
echo ""
echo "=== Training MLM transformer ($(date)) ==="
python scripts/train_mlm.py \
    --data          "${DATA}" \
    --vocab         "${VOCAB}" \
    --max-length    "${MAX_LENGTH}" \
    --d-model       "${D_MODEL}" \
    --nhead         "${NHEAD}" \
    --num-layers    "${NUM_LAYERS}" \
    --dim-feedforward "${DIM_FEEDFORWARD}" \
    --dropout       "${DROPOUT}" \
    --lr            "${LR}" \
    --batch-size    "${BATCH_SIZE}" \
    --epochs        "${EPOCHS}" \
    --mask-prob     "${MASK_PROB}" \
    --mean-span     "${MEAN_SPAN}" \
    --max-span      "${MAX_SPAN}" \
    --val-split     "${VAL_SPLIT}" \
    --warmup-steps  "${WARMUP_STEPS}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --num-workers   "${NUM_WORKERS}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output        "${OUTPUT}"

echo ""
echo "Model weights: ${OUTPUT}"
echo "Model config:  ${OUTPUT%.pt}.json"
echo "Best checkpoint: ${CHECKPOINT_DIR}/best.pt"

# ---- collect environment info (on compute node with GPU) ----
ENV_JSON="${OUTPUT%.pt}_env.json"
echo ""
echo "=== Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${ENV_JSON}"

# ---- generate training plot ----
FIGURE="${OUTPUT%.pt}_training_loss.pdf"
echo ""
echo "=== Generating training plot ($(date)) ==="
python scripts/plot_training.py \
    --config  "${OUTPUT%.pt}.json" \
    --output  "${FIGURE%.pdf}" \
    --format  pdf

# ---- generate report ----
REPORT="${OUTPUT%.pt}_report.md"
echo ""
echo "=== Generating report ($(date)) ==="
python scripts/generate_mlm_report.py \
    --config  "${OUTPUT%.pt}.json" \
    --env     "${ENV_JSON}" \
    --figure  "${FIGURE}" \
    --output  "${REPORT}"

echo ""
echo "Training plot: ${FIGURE}"
echo "Report:        ${REPORT}"

# ---- version-track model artifacts with DVC ----
echo ""
echo "=== Tracking model artifacts with DVC ($(date)) ==="
dvc add "${OUTPUT}" "${OUTPUT%.pt}.json"

git add "${OUTPUT}.dvc" "${OUTPUT%.pt}.json.dvc" models/.gitignore
git commit -m "[DVC] Track MLM model weights and config (run ${RUN_ID})"
dvc push

echo "=== Job finished on $(date) ==="

deactivate