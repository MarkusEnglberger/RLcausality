#!/bin/bash
# Quick script to run SFT pipeline locally (not on SLURM)
# Usage: bash scripts/run_sft_pipeline.sh

set -e  # Exit on error

echo "=========================================="
echo "SFT Training Pipeline - DeepSeek 32B"
echo "=========================================="
echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data/processed/sft_deepseek
mkdir -p models/deepseek_sft_lora_4bit

# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache
export WANDB_PROJECT="corr2cause-sft"

# Step 1: Prepare dataset
echo "Step 1: Preparing SFT dataset from GPT predictions..."
python scripts/prepare_sft_data.py
echo "✓ Dataset preparation complete"
echo ""

# Step 2: Train model
echo "Step 2: Starting SFT training with LoRA and 4-bit quantization..."
echo "This may take 1-2 hours on H100..."
python scripts/train_sft_lora.py configs/sft_config_deepseek.yaml
echo "✓ Training complete"
echo ""

echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Model saved to: ./models/deepseek_sft_lora_4bit"
echo ""
echo "To evaluate the model, update configs/evaluation_config.yaml:"
echo "  model_path: './models/deepseek_sft_lora_4bit'"
echo "Then run: python scripts/evaluate_model_simple.py"
