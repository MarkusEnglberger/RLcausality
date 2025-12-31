#!/bin/bash
# Convenience script to evaluate all three model families:
# 1. Qwen2.5-1.5B (SFT + GRPO)
# 2. Phi-3.5-mini-instruct (SFT + GRPO)
# 3. DeepSeek-R1-Distill-Qwen-7B (SFT + GRPO)

echo "==================================================================="
echo "Submitting evaluation jobs for all models"
echo "Note: Evaluating on default subset (50 samples per model)"
echo "==================================================================="
echo ""

# Array to store job IDs
declare -a JOB_IDS
declare -a JOB_NAMES

# Function to submit evaluation job if model exists
submit_eval_job() {
    local model_path=$1
    local model_type=$2
    local model_name=$3

    if [ -d "${model_path}" ]; then
        echo "Submitting ${model_name} evaluation..."
        JOB_ID=$(sbatch --parsable submit_evaluation.slurm "${model_path}" "${model_type}")
        JOB_IDS+=("${JOB_ID}")
        JOB_NAMES+=("${model_name}")
        echo "${model_name} evaluation job submitted: ${JOB_ID}"
    else
        echo "WARNING: ${model_name} not found at ${model_path} - skipping"
    fi
    echo ""
}

# Evaluate Qwen2.5-1.5B models
echo "=== Qwen2.5-1.5B Models ==="
submit_eval_job "./models/sft_model" "sft" "Qwen2.5-1.5B-SFT"

# Find latest GRPO checkpoint for Qwen
QWEN_GRPO_PATH="./models/grpo_model"
if [ -d "${QWEN_GRPO_PATH}" ]; then
    LATEST_CHECKPOINT=$(ls -d ${QWEN_GRPO_PATH}/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        submit_eval_job "${LATEST_CHECKPOINT}" "grpo" "Qwen2.5-1.5B-GRPO"
    else
        echo "WARNING: No Qwen GRPO checkpoints found"
        echo ""
    fi
fi

# Evaluate Phi-3.5-mini models
echo "=== Phi-3.5-mini Models ==="
submit_eval_job "./models/phi3_sft_model/checkpoint-900" "sft" "Phi-3.5-mini-SFT"

# Find latest GRPO checkpoint for Phi-3
PHI3_GRPO_PATH="./models/phi3_grpo_model"
if [ -d "${PHI3_GRPO_PATH}" ]; then
    LATEST_CHECKPOINT=$(ls -d ${PHI3_GRPO_PATH}/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        submit_eval_job "${LATEST_CHECKPOINT}" "grpo" "Phi-3.5-mini-GRPO"
    else
        echo "WARNING: No Phi-3 GRPO checkpoints found"
        echo ""
    fi
fi

# Evaluate DeepSeek-R1-Distill-Qwen-7B models
echo "=== DeepSeek-R1-Distill-Qwen-7B Models ==="
submit_eval_job "./models/deepseek_sft_model" "sft" "DeepSeek-R1-7B-SFT"

# Find latest GRPO checkpoint for DeepSeek
DEEPSEEK_GRPO_PATH="./models/deepseek_grpo_model"
if [ -d "${DEEPSEEK_GRPO_PATH}" ]; then
    LATEST_CHECKPOINT=$(ls -d ${DEEPSEEK_GRPO_PATH}/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        submit_eval_job "${LATEST_CHECKPOINT}" "grpo" "DeepSeek-R1-7B-GRPO"
    else
        echo "WARNING: No DeepSeek GRPO checkpoints found"
        echo ""
    fi
fi

# Summary
echo "==================================================================="
echo "Evaluation jobs submitted!"
echo "==================================================================="
echo ""
echo "Jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  ${JOB_NAMES[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: logs/eval_*.out"
echo "Results will be saved in: evaluation_results/"
echo ""
echo "==================================================================="
