#!/bin/bash
#SBATCH --job-name=sft_qwen1.5b
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=tue.gpu2.q
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load required modules (adjust based on your HPC environment)
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

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

# Create logs directory
mkdir -p logs

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache
export WANDB_PROJECT="corr2cause-sft-qwen"
# Let SLURM manage CUDA_VISIBLE_DEVICES automatically

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS"
echo "Working directory: $(pwd)"
nvidia-smi

# Run preprocessing (if not already done)
if [ ! -d "./data/processed/sft" ]; then
    echo "Running data preprocessing..."
    python scripts/data_preprocessing.py
fi

# Run SFT training with 2 GPUs (optimized for Qwen2.5-1.5B)
echo "Starting SFT training with Qwen2.5-1.5B..."
torchrun --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_sft.py \
    configs/sft_config.yaml \
    --run_name "sft-qwen1.5b-${SLURM_JOB_ID}"

echo "Job finished at: $(date)"
