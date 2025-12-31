#!/bin/bash
#SBATCH --job-name=eval_corr2cause
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=tue.gpu2.q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load required modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

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

# Create output directories
mkdir -p logs
mkdir -p evaluation_results

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
nvidia-smi

# Parse command line arguments
MODEL_PATH=${1:-"./models/sft_model"}
MODEL_TYPE=${2:-"sft"}

echo ""
echo "====================================="
echo "Evaluating ${MODEL_TYPE} model"
echo "Model path: ${MODEL_PATH}"
echo "====================================="
echo ""

# Run evaluation on test subset using local preprocessed data
# Note: --use_local_data flag avoids corrupted HuggingFace CSV files
python scripts/evaluate_model.py \
    --model_path "${MODEL_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --subsets default \
    --output_dir ./evaluation_results \
    --use_4bit \
    --batch_size 4 \
    --max_new_tokens 512 \
    --max_samples 50 \
    --use_local_data \
    --show_samples 40

echo ""
echo "Job finished at: $(date)"
