#!/bin/bash
#SBATCH --job-name=preprocess_deepseek
#SBATCH --output=logs/preprocess_deepseek_%j.out
#SBATCH --error=logs/preprocess_deepseek_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load required modules
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup_hpc.sh first:"
    echo "  bash setup_hpc.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Create necessary directories
mkdir -p data/cache
mkdir -p data/processed
mkdir -p logs

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Run preprocessing for DeepSeek-R1-Distill-Qwen-7B
echo "Starting data preprocessing for DeepSeek-R1-Distill-Qwen-7B..."
python scripts/data_preprocessing_deepseek.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --stage both

echo "Job finished at: $(date)"
