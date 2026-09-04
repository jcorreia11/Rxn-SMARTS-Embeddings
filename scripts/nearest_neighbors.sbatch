#!/bin/bash
#SBATCH --job-name=nearest_neighbors
#SBATCH --account=f202607628cpcaa1x
#SBATCH --partition=normal-x86
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

# 16G previously OOM-risked here too: load_reaction_groups() in
# nearest_neighbors.py reads the full validated_smarts.csv (~361k rows) into
# a pandas DataFrame plus a dict, same as similarity_correlation.py — see
# that sbatch script's memory note. Bumped preemptively before first run.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- config ----
# Override at submission time with --export=ALL,VAR=value
# Example: pick a specific query index
#   sbatch --export=ALL,QUERY_INDEX=42 nearest_neighbors.sbatch
# Example: use a random query with a fixed seed
#   sbatch --export=ALL,QUERY_MODE=random,SEED=7 nearest_neighbors.sbatch
#
# MODEL_SIZE (small/medium/large) also shapes the default EMBEDDINGS path
# and the results/final/3d-nearest-neighbors/${MODEL_SIZE}/ mirror below.
MODEL_SIZE="${MODEL_SIZE:-unspecified}"
EMBEDDINGS="${EMBEDDINGS:-data/embeddings/${MODEL_SIZE}/embeddings.npy}"
SMARTS_FILE="${SMARTS_FILE:-data/embeddings/${MODEL_SIZE}/smarts.txt}"
TOP_K="${TOP_K:-5}"
METRIC="${METRIC:-cosine}"
SEED="${SEED:-42}"

# Query mode: "index" (default), "random", or "smarts"
QUERY_MODE="${QUERY_MODE:-index}"
QUERY_INDEX="${QUERY_INDEX:-0}"
QUERY_SMARTS="${QUERY_SMARTS:-}"
# Optional label for ad-hoc example queries (e.g. "ec1_query") so they can
# be mirrored alongside the main run without overwriting results.json.
QUERY_LABEL="${QUERY_LABEL:-}"

# Derive a run ID from the job ID for unique output paths
RUN_ID="nn_${SLURM_JOB_ID}"
OUTPUT_JSON="${OUTPUT_JSON:-results/${RUN_ID}.json}"
OUTPUT_ENV="${OUTPUT_ENV:-results/${RUN_ID}_env.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-results/${RUN_ID}_report.md}"

mkdir -p results

echo ""
echo "Embeddings:   ${EMBEDDINGS}"
echo "SMARTS file:  ${SMARTS_FILE}"
echo "Top K:        ${TOP_K}"
echo "Metric:       ${METRIC}"
echo "Query mode:   ${QUERY_MODE}"
if [ "${QUERY_MODE}" = "index" ]; then
    echo "Query index:  ${QUERY_INDEX}"
elif [ "${QUERY_MODE}" = "random" ]; then
    echo "Random seed:  ${SEED}"
elif [ "${QUERY_MODE}" = "smarts" ]; then
    echo "Query SMARTS: ${QUERY_SMARTS}"
fi
echo "Results JSON: ${OUTPUT_JSON}"
echo "Report:       ${OUTPUT_REPORT}"
echo "Model size:   ${MODEL_SIZE}"
echo "Query label:  ${QUERY_LABEL:-<none>}"

# ---- collect environment info ----
echo ""
echo "=== Collecting environment info ($(date)) ==="
python scripts/collect_env.py --output "${OUTPUT_ENV}"

# ---- build query flag ----
if [ "${QUERY_MODE}" = "random" ]; then
    QUERY_FLAG="--random-query --seed ${SEED}"
elif [ "${QUERY_MODE}" = "smarts" ]; then
    QUERY_FLAG="--query-smarts ${QUERY_SMARTS}"
else
    QUERY_FLAG="--query-index ${QUERY_INDEX}"
fi

# ---- run nearest-neighbor search ----
echo ""
echo "=== Running nearest-neighbor search ($(date)) ==="
python scripts/nearest_neighbors.py \
    --embeddings "${EMBEDDINGS}" \
    --smarts     "${SMARTS_FILE}" \
    --top-k      "${TOP_K}" \
    --metric     "${METRIC}" \
    --output     "${OUTPUT_JSON}" \
    ${QUERY_FLAG}

# ---- generate report ----
echo ""
echo "=== Generating report ($(date)) ==="
python scripts/generate_nn_report.py \
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
echo "=== Mirroring artifacts into results/final/3d-nearest-neighbors/${MODEL_SIZE}/ ($(date)) ==="
MIRROR_DIR="results/final/3d-nearest-neighbors/${MODEL_SIZE}"
mkdir -p "${MIRROR_DIR}"
if [ -n "${QUERY_LABEL}" ]; then
    cp "${OUTPUT_JSON}"   "${MIRROR_DIR}/${QUERY_LABEL}.json"
    cp "${OUTPUT_ENV}"    "${MIRROR_DIR}/${QUERY_LABEL}_env.json"
    cp "${OUTPUT_REPORT}" "${MIRROR_DIR}/${QUERY_LABEL}_report.md"
else
    cp "${OUTPUT_JSON}"   "${MIRROR_DIR}/results.json"
    cp "${OUTPUT_ENV}"    "${MIRROR_DIR}/env.json"
    cp "${OUTPUT_REPORT}" "${MIRROR_DIR}/report.md"
fi
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${MIRROR_DIR}/" 2>/dev/null || true
cp "/projects/F202607628CPCAA1/joao.correia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${MIRROR_DIR}/" 2>/dev/null || true

echo "=== Job finished on $(date) ==="

deactivate