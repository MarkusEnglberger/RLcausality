#!/bin/bash
# Complete pipeline for training Phi-3.5-Mini on Corr2Cause dataset
# This script submits jobs in the correct order with dependencies

echo "=========================================================================="
echo "Phi-3.5-Mini Training Pipeline for Causal Reasoning"
echo "=========================================================================="
echo ""
echo "This will submit 3 jobs in sequence:"
echo "  1. Data preprocessing (1 hour, CPU only)"
echo "  2. SFT training (4 hours, 2x GPUs)"
echo "  3. GRPO training (8 hours, 2x GPUs)"
echo ""
echo "Total expected time: ~13 hours"
echo "=========================================================================="
echo ""

# Submit preprocessing job
echo "[1/3] Submitting data preprocessing job..."
PREPROCESS_JOB=$(sbatch --parsable submit_preprocessing_phi3.slurm)
echo "      Job ID: ${PREPROCESS_JOB}"

# Submit SFT job (depends on preprocessing)
echo "[2/3] Submitting SFT training job (will wait for preprocessing)..."
SFT_JOB=$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB} submit_sft_phi3.slurm)
echo "      Job ID: ${SFT_JOB}"

# Submit GRPO job (depends on SFT)
echo "[3/3] Submitting GRPO training job (will wait for SFT)..."
GRPO_JOB=$(sbatch --parsable --dependency=afterok:${SFT_JOB} submit_grpo_phi3.slurm)
echo "      Job ID: ${GRPO_JOB}"

echo ""
echo "=========================================================================="
echo "All jobs submitted successfully!"
echo "=========================================================================="
echo ""
echo "Job IDs:"
echo "  Preprocessing: ${PREPROCESS_JOB}"
echo "  SFT:           ${SFT_JOB}"
echo "  GRPO:          ${GRPO_JOB}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/preprocess_phi3_${PREPROCESS_JOB}.out"
echo "  tail -f logs/sft_phi3_${SFT_JOB}.out"
echo "  tail -f logs/grpo_phi3_${GRPO_JOB}.out"
echo ""
echo "Check logs in: logs/"
echo "Models will be saved in: models/phi3_sft_model and models/phi3_grpo_model"
echo "=========================================================================="
