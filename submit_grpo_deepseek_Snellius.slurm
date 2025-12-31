#!/bin/bash
#SBATCH --job-name=grpo_deepseek7b
#SBATCH --output=logs/grpo_deepseek_%j.out
#SBATCH --error=logs/grpo_deepseek_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=180G

# Load required modules
module purge
module load 2025  # Load the 2023 software stack first
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0

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

# Create logs directory
mkdir -p logs

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache
export WANDB_PROJECT="corr2cause-grpo-deepseek"

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS"
echo "Working directory: $(pwd)"
nvidia-smi

# Run preprocessing for GRPO (if not already done)
if [ ! -d "./data/processed/grpo_deepseek" ]; then
    echo "Running data preprocessing for GRPO..."
    python scripts/data_preprocessing_deepseek.py --stage grpo
fi

# Run GRPO training with 2 GPUs using DeepSpeed ZeRO-2
echo "Starting GRPO training with DeepSeek-R1-Distill-Qwen-7B..."
echo "Note: By default, training directly from base model (no SFT needed)"
echo "Using DeepSpeed ZeRO Stage 2 (ZeRO-3 causes OOM during generation)"
echo "Check configs/grpo_config_deepseek.yaml for model_name_or_path setting"

accelerate launch \
    --config_file configs/accelerate_deepspeed_zero2.yaml \
    scripts/train_grpo.py \
    configs/grpo_config_deepseek.yaml \
    --run_name "grpo-deepseek7b-${SLURM_JOB_ID}"

echo "Job finished at: $(date)"
