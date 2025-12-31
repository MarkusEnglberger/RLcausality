#!/bin/bash
# Convenience script to evaluate both SFT and GRPO models

echo "==================================================================="
echo "Submitting evaluation jobs for both SFT and GRPO models"
echo "Note: Evaluating 150 samples per subset (default, refactorization, paraphrasing)"
echo "==================================================================="
echo ""

# Evaluate SFT model
echo "Submitting SFT model evaluation..."
SFT_JOB_ID=$(sbatch --parsable submit_evaluation.slurm ./models/sft_model sft)
echo "SFT evaluation job submitted: ${SFT_JOB_ID}"

# Find the latest GRPO checkpoint
GRPO_MODEL_PATH="./models/grpo_model"
if [ -d "${GRPO_MODEL_PATH}" ]; then
    # Find the highest numbered checkpoint
    LATEST_CHECKPOINT=$(ls -d ${GRPO_MODEL_PATH}/checkpoint-* 2>/dev/null | sort -V | tail -1)

    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo ""
        echo "Submitting GRPO model evaluation..."
        echo "Using checkpoint: ${LATEST_CHECKPOINT}"
        GRPO_JOB_ID=$(sbatch --parsable submit_evaluation.slurm "${LATEST_CHECKPOINT}" grpo)
        echo "GRPO evaluation job submitted: ${GRPO_JOB_ID}"
    else
        echo ""
        echo "WARNING: No GRPO checkpoints found in ${GRPO_MODEL_PATH}"
        echo "Please wait for GRPO training to complete and create checkpoints"
    fi
else
    echo ""
    echo "WARNING: GRPO model directory not found at ${GRPO_MODEL_PATH}"
    echo "Please wait for GRPO training to complete"
fi

echo ""
echo "==================================================================="
echo "Evaluation jobs submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: logs/eval_*.out"
echo "Results will be saved in: evaluation_results/"
echo "==================================================================="
