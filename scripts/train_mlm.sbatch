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

# ---- ensure cu118 torch (pip ignores uv index sources) ----
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---- config ----
# Override at submission time if needed:
#   sbatch --export=ALL,EPOCHS=50,D_MODEL=256 train_mlm.sbatch
DATA="${DATA:-data/processed/validated_smarts.csv}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
MAX_LENGTH="${MAX_LENGTH:-128}"
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-512}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
MASK_PROB="${MASK_PROB:-0.15}"

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
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output        "${OUTPUT}"

echo ""
echo "Model weights: ${OUTPUT}"
echo "Model config:  ${OUTPUT%.pt}.json"
echo "Best checkpoint: ${CHECKPOINT_DIR}/best.pt"
echo "=== Job finished on $(date) ==="

deactivate