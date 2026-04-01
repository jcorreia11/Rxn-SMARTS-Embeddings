#!/bin/bash
#SBATCH --job-name=smartrxn_dvc
#SBATCH --account=f202508983cpcaa0x
#SBATCH --partition=normal-x86
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.out
#SBATCH --error=/projects/F202508983CPCAA0/jcorreia/logs/%x_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202508983CPCAA0/jcorreia/work/SmartRxnEmbeddings

source ../smartrxn_env/bin/activate

# ---- sanity check ----
echo "Python executable: $(which python)"
python -V
python -c "import sys; print('sys.path ok, site:', sys.path[0])"
dvc version

# ---- run ----
RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "Run ID: ${RUN_ID}"

dvc repro
dvc push

deactivate
echo "=== Job finished on $(date) ==="