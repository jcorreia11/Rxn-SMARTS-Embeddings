#!/bin/bash
#SBATCH --job-name=extract_embeddings
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# ---- ensure cu124 torch (driver supports CUDA 12.4; force-reinstall to replace cu130 build) ----
pip install --force-reinstall --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---- config ----
# Override at submission time:
#   sbatch --export=ALL,WEIGHTS=models/smarts_transformer_<RUN_ID>.pt extract_embeddings.sbatch
WEIGHTS="${WEIGHTS:-models/smarts_transformer.pt}"
CONFIG="${CONFIG:-${WEIGHTS%.pt}.json}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
DATA="${DATA:-data/processed/validated_smarts.csv}"
OUTPUT="${OUTPUT:-data/embeddings/reaction_embeddings.npy}"
SMARTS_OUTPUT="${SMARTS_OUTPUT:-data/embeddings/reaction_smarts.txt}"
POOLING="${POOLING:-cls}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo ""
echo "Weights:       ${WEIGHTS}"
echo "Config:        ${CONFIG}"
echo "Vocab:         ${VOCAB}"
echo "Data:          ${DATA}"
echo "Output:        ${OUTPUT}"
echo "SMARTS output: ${SMARTS_OUTPUT}"
echo "Pooling:       ${POOLING}"
echo "Batch size:    ${BATCH_SIZE}"

# ---- extract ----
echo ""
echo "=== Extracting embeddings ($(date)) ==="
python scripts/extract_embeddings.py \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --vocab         "${VOCAB}" \
    --data          "${DATA}" \
    --output        "${OUTPUT}" \
    --smarts-output "${SMARTS_OUTPUT}" \
    --pooling       "${POOLING}" \
    --batch-size    "${BATCH_SIZE}"

echo ""
echo "Embeddings: ${OUTPUT}"
echo "SMARTS list: ${SMARTS_OUTPUT}"
echo "=== Job finished on $(date) ==="

deactivate