#!/bin/bash
# Complete training pipeline for DeepSeek-R1-Distill-Qwen-7B
# This script runs the full pipeline: preprocessing -> SFT -> GRPO

set -e  # Exit on error

echo "==================================================================="
echo "DeepSeek-R1-Distill-Qwen-7B Training Pipeline"
echo "This will run: Preprocessing -> SFT Training -> GRPO Training"
echo "==================================================================="
echo ""

# Step 1: Submit preprocessing job
echo "Step 1/3: Submitting data preprocessing job..."
PREPROCESS_JOB_ID=$(sbatch --parsable submit_preprocessing_deepseek.slurm)
echo "Preprocessing job submitted: ${PREPROCESS_JOB_ID}"
echo ""

# Step 2: Submit SFT training (depends on preprocessing)
echo "Step 2/3: Submitting SFT training job (depends on preprocessing)..."
SFT_JOB_ID=$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} submit_sft_deepseek.slurm)
echo "SFT training job submitted: ${SFT_JOB_ID}"
echo ""

# Step 3: Submit GRPO training (depends on SFT)
echo "Step 3/3: Submitting GRPO training job (depends on SFT)..."
GRPO_JOB_ID=$(sbatch --parsable --dependency=afterok:${SFT_JOB_ID} submit_grpo_deepseek.slurm)
echo "GRPO training job submitted: ${GRPO_JOB_ID}"
echo ""

echo "==================================================================="
echo "All jobs submitted successfully!"
echo "==================================================================="
echo ""
echo "Job chain:"
echo "  1. Preprocessing: ${PREPROCESS_JOB_ID}"
echo "  2. SFT Training:  ${SFT_JOB_ID} (waits for ${PREPROCESS_JOB_ID})"
echo "  3. GRPO Training: ${GRPO_JOB_ID} (waits for ${SFT_JOB_ID})"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f logs/preprocess_deepseek_${PREPROCESS_JOB_ID}.out"
echo "  tail -f logs/sft_deepseek_${SFT_JOB_ID}.out"
echo "  tail -f logs/grpo_deepseek_${GRPO_JOB_ID}.out"
echo ""
echo "Models will be saved in:"
echo "  ./models/deepseek_sft_model   (after SFT)"
echo "  ./models/deepseek_grpo_model  (after GRPO)"
echo ""
echo "==================================================================="
