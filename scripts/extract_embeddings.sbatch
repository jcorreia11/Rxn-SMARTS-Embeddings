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
# Override at submission time:
#   sbatch --export=ALL,WEIGHTS=models/smarts_transformer_<RUN_ID>.pt extract_embeddings.sbatch
WEIGHTS="${WEIGHTS:-models/smarts_transformer.pt}"
CONFIG="${CONFIG:-${WEIGHTS%.pt}.json}"
VOCAB="${VOCAB:-data/processed/vocab.json}"
DATA="${DATA:-data/processed/validated_smarts.csv}"
# Label only (small/medium/large) — also shapes the default OUTPUT paths
# below and the results/final/2-extract-embeddings/${MODEL_SIZE}/ mirror.
MODEL_SIZE="${MODEL_SIZE:-unspecified}"
OUTPUT="${OUTPUT:-data/embeddings/${MODEL_SIZE}/embeddings.npy}"
SMARTS_OUTPUT="${SMARTS_OUTPUT:-data/embeddings/${MODEL_SIZE}/smarts.txt}"
POOLING="${POOLING:-cls}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo ""
echo "Weights:       ${WEIGHTS}"
echo "Config:        ${CONFIG}"
echo "Vocab:         ${VOCAB}"
echo "Data:          ${DATA}"
echo "Model size:    ${MODEL_SIZE}"
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

# ${OUTPUT} and ${SMARTS_OUTPUT} are gitignored, local-only artifacts — not
# versioned. Copy them off this machine yourself (rsync/scp) if you need
# them elsewhere; nothing here backs them up.

# ---- mirror into results/final/ for easy browsing ----
# (local convenience copy only, gitignored — data/embeddings/ above is the
# only copy either way, this just makes it easier to browse alongside
# everything else)
echo ""
echo "=== Mirroring artifacts into results/final/2-extract-embeddings/${MODEL_SIZE}/ ($(date)) ==="
MIRROR_DIR="results/final/2-extract-embeddings/${MODEL_SIZE}"
mkdir -p "${MIRROR_DIR}"
cp "${OUTPUT}" "${SMARTS_OUTPUT}" "${MIRROR_DIR}/"
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202508983CPCAA0/jcorreia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate