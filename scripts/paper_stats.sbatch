#!/bin/bash
#SBATCH --job-name=paper_stats
#SBATCH --account=f202607628cpcaa1x
#SBATCH --partition=normal-x86
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

# Aggregates every paper/*_stats.json used by PAPER_TEMPLATE_JCIM.md, then
# mirrors them (+ paper/figures/) into results/final/4-paper/. Run this last,
# once every other phase's results/final/ mirror already exists on this
# machine (see hpc-runbook.md) — most of these scripts read directly from
# results/final/<phase>/<size>/.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- dataset & tokenizer stats ----
echo ""
echo "=== 1/9: dataset_stats.py ($(date)) ==="
python paper/dataset_stats.py

echo ""
echo "=== 2/9: tokenizer_stats.py ($(date)) ==="
python paper/tokenizer_stats.py

echo ""
echo "=== 3/9: dataset_tokenizer_results.py (needs tokenizer_stats.json) ($(date)) ==="
python paper/dataset_tokenizer_results.py

# ---- model / training stats ----
echo ""
echo "=== 4/9: model_architecture_stats.py ($(date)) ==="
python paper/model_architecture_stats.py

echo ""
echo "=== 5/9: mlm_stats.py ($(date)) ==="
python paper/mlm_stats.py

echo ""
echo "=== 6/9: training_stats.py ($(date)) ==="
python paper/training_stats.py

# ---- downstream evaluation stats ----
echo ""
echo "=== 7/9: pooling_ablation_stats.py ($(date)) ==="
python paper/pooling_ablation_stats.py

echo ""
echo "=== 8/9: ec_classifier_stats.py ($(date)) ==="
python paper/ec_classifier_stats.py

echo ""
echo "=== 9/9: similarity_correlation_stats.py ($(date)) ==="
python paper/similarity_correlation_stats.py

# ---- commit results to git ----
echo ""
echo "=== Committing paper/ stats to git ($(date)) ==="
git add paper/*.json
git commit -m "[results] Paper statistics (job ${SLURM_JOB_ID})"

# ---- mirror into results/final/4-paper/ for easy browsing ----
echo ""
echo "=== Mirroring into results/final/4-paper/ ($(date)) ==="
python paper/sync_to_results_final.py

echo "=== Job finished on $(date) ==="

deactivate