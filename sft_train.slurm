#!/bin/bash
#SBATCH --job-name=sft-lora-4bit
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G

# Load required modules
module purge
module load 2025
module load CUDA/12.8.0
module load Python/3.13.1-GCCcore-14.2.0

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p data/processed/sft_deepseek
mkdir -p models/deepseek_sft_lora_4bit

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache
export WANDB_PROJECT="corr2cause-sft"

# Print job information
echo "=========================================="
echo "SFT Training Job - DeepSeek 32B with LoRA and 4-bit Quantization"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS"
echo "Working directory: $(pwd)"
echo ""
echo "GPU Information:"
nvidia-smi
echo "=========================================="
echo ""

# Step 1: Prepare SFT dataset from GPT predictions
echo "Step 1: Preparing SFT dataset from correct GPT reasoning traces..."
python scripts/prepare_sft_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed"
    exit 1
fi
echo ""

# Step 2: Run SFT training with LoRA and 4-bit quantization
echo "Step 2: Starting SFT training..."
python scripts/train_sft_lora.py configs/sft_config_deepseek.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "Job finished successfully at: $(date)"
echo "=========================================="
echo ""
echo "Model saved to: ./models/deepseek_sft_lora_4bit"
echo "To use the finetuned model, load it with:"
echo "  from peft import PeftModel"
echo "  model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', ...)"
echo "  model = PeftModel.from_pretrained(model, './models/deepseek_sft_lora_4bit')"
