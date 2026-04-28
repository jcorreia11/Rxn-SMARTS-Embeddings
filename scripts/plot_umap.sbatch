#!/bin/bash
#SBATCH --job-name=plot_umap
#SBATCH --account=F202508983CPCAA0G
#SBATCH --partition=normal-a100-40
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

echo "Python executable: $(which python)"
python -V

# ---- config ----
# Override at submission time with --export=ALL,VAR=value
EMBEDDINGS="${EMBEDDINGS:-data/embeddings/reaction_embeddings.npy}"
SMARTS_FILE="${SMARTS_FILE:-data/embeddings/reaction_smarts.txt}"
EC_DEPTH="${EC_DEPTH:-1}"
FORMAT="${FORMAT:-pdf}"
N_NEIGHBORS="${N_NEIGHBORS:-15}"
MIN_DIST="${MIN_DIST:-0.1}"
METRIC="${METRIC:-cosine}"
RANDOM_STATE="${RANDOM_STATE:-42}"
POINT_SIZE="${POINT_SIZE:-0.8}"
ALPHA="${ALPHA:-0.4}"
DPI="${DPI:-300}"

# Output path encodes the EC depth so depth-1 and depth-2 figures don't overwrite each other.
EMBEDDINGS_STEM=$(basename "${EMBEDDINGS}" .npy)
OUTPUT="${OUTPUT:-results/figures/umap_embedding_space_d${EC_DEPTH}}"

# Reuse cached 2-D coordinates — UMAP only depends on embeddings, not on EC labels,
# so changing EC_DEPTH does not require re-running the reduction.
UMAP_CACHE="results/figures/${EMBEDDINGS_STEM}_umap_coords.npy"

mkdir -p results/figures

echo ""
echo "Embeddings:    ${EMBEDDINGS}"
echo "SMARTS file:   ${SMARTS_FILE}"
echo "EC depth:      ${EC_DEPTH}"
echo "UMAP cache:    ${UMAP_CACHE}"
echo "Output:        ${OUTPUT}.${FORMAT}"
echo "n_neighbors:   ${N_NEIGHBORS}"
echo "min_dist:      ${MIN_DIST}"
echo "metric:        ${METRIC}"
echo "random_state:  ${RANDOM_STATE}"
echo "point_size:    ${POINT_SIZE}"
echo "alpha:         ${ALPHA}"

# ---- run ----
echo ""
echo "=== Running UMAP ($(date)) ==="
python scripts/plot_umap.py \
    --embeddings    "${EMBEDDINGS}" \
    --smarts        "${SMARTS_FILE}" \
    --umap-cache    "${UMAP_CACHE}" \
    --output        "${OUTPUT}" \
    --format        "${FORMAT}" \
    --dpi           "${DPI}" \
    --n-neighbors   "${N_NEIGHBORS}" \
    --min-dist      "${MIN_DIST}" \
    --metric        "${METRIC}" \
    --random-state  "${RANDOM_STATE}" \
    --point-size    "${POINT_SIZE}" \
    --alpha         "${ALPHA}" \
    --ec-depth      "${EC_DEPTH}"

echo ""
echo "Figure:     ${OUTPUT}.${FORMAT}"
echo "UMAP cache: ${UMAP_CACHE}"

# ---- commit results to git ----
echo ""
echo "=== Committing results to git ($(date)) ==="
git add "${OUTPUT}.${FORMAT}"
git commit -m "[results] UMAP embedding space depth=${EC_DEPTH} (job ${SLURM_JOB_ID})"

echo "=== Job finished on $(date) ==="

deactivate