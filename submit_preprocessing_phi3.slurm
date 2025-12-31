#!/bin/bash
#SBATCH --job-name=preprocess_phi3
#SBATCH --output=logs/preprocess_phi3_%j.out
#SBATCH --error=logs/preprocess_phi3_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load required modules
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup_hpc.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Create directories
mkdir -p logs
mkdir -p data/processed
mkdir -p data/cache

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

echo ""
echo "======================================"
echo "Preprocessing data for Phi-3.5-Mini"
echo "======================================"
echo ""

# Run preprocessing for both SFT and GRPO
python scripts/data_preprocessing_phi3.py \
    --mode both \
    --model_name microsoft/Phi-3.5-mini-instruct \
    --output_dir ./data/processed \
    --cache_dir ./data/cache

echo ""
echo "Job finished at: $(date)"
