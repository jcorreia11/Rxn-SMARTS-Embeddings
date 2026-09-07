#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=f202607628cpcaa1x
#SBATCH --partition=normal-x86
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.out
#SBATCH --error=/projects/F202607628CPCAA1/joao.correia/logs/%x_%j.err

# Runs the full preprocessing chain (no pipeline tool — each step is cheap
# and deterministic, plain sequential python):
#   raw CSVs -> load_data -> validate_smarts -> build_vocab -> sentencepiece_tokenizer
# Outputs are local, gitignored artifacts (data/raw/*, data/processed/*) —
# not versioned or auto-synced between machines.

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd /projects/F202607628CPCAA1/joao.correia/work/Rxn-SMARTS-Embeddings

source ../smartrxn_env/bin/activate

echo "Python executable: $(which python)"
python -V

# ---- step 1/4: load & deduplicate raw RetroRules CSVs ----
echo ""
echo "=== Step 1/4: load_data.py ($(date)) ==="
python src/rxn_smarts_embeddings/preprocessing/load_data.py

# ---- step 2/4: RDKit validation + reaction_group join ----
echo ""
echo "=== Step 2/4: validate_smarts.py ($(date)) ==="
python src/rxn_smarts_embeddings/preprocessing/validate_smarts.py

# ---- step 3/4: build rule-based tokenizer vocabulary ----
echo ""
echo "=== Step 3/4: build_vocab.py ($(date)) ==="
python src/rxn_smarts_embeddings/tokenization/build_vocab.py

# ---- step 4/4: train SentencePiece BPE baseline tokenizer ----
echo ""
echo "=== Step 4/4: sentencepiece_tokenizer.py ($(date)) ==="
python src/rxn_smarts_embeddings/tokenization/sentencepiece_tokenizer.py

echo ""
echo "Outputs:"
echo "  data/processed/clean_smarts.txt"
echo "  data/processed/reaction_groups.csv"
echo "  data/processed/validated_smarts.csv"
echo "  data/processed/vocab.json"
echo "  data/processed/sp_tokenizer.model / .vocab"

echo "=== Job finished on $(date) ==="

deactivate