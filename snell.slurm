#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G

# Load required modules
module purge
module load 2025
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
# Using 2023 stack for better library compatibility

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

# CUDA configuration
#export CUDA_HOME=/sw/arch/RHEL9/EB_production/2023/software/CUDA/12.1.1
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# DeepSpeed configuration - disable all ops compilation
#export DS_BUILD_OPS=0
#export DS_SKIP_CUDA_CHECK=1

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS"
echo "Working directory: $(pwd)"
nvidia-smi


python scripts/evaluate_model_simple.py


torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
    scripts/train_grpo.py \
    configs/grpo_config_deepseek.yaml \
    --run_name "grpo-deepseek7b-torchrun-${SLURM_JOB_ID}"

echo "Job finished at: $(date)"