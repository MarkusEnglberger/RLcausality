#!/bin/bash
# Script to run GRPO training on multiple GPUs (for local/interactive use)

# Number of GPUs to use (optimized for Qwen2.5-1.5B)
NUM_GPUS=2  # Change this to your available GPU count

# Using torchrun (PyTorch native, simple)
torchrun --nproc_per_node=$NUM_GPUS \
    scripts/train_grpo.py \
    configs/grpo_config.yaml \
    --wandb_project "corr2cause-grpo-qwen" \
    --run_name "grpo-qwen1.5b-local"

# Alternative: Using accelerate (recommended for complex setups)
# First run: accelerate config
# Then run:
# accelerate launch --num_processes=$NUM_GPUS \
#     scripts/train_grpo.py \
#     configs/grpo_config.yaml \
#     --wandb_project "corr2cause-grpo-qwen" \
#     --run_name "grpo-qwen1.5b-local"

# Alternative: Using DeepSpeed for ZeRO optimization
# deepspeed --num_gpus=$NUM_GPUS \
#     scripts/train_grpo.py \
#     configs/grpo_config.yaml \
#     --deepspeed configs/deepspeed_config.json \
#     --wandb_project "corr2cause-grpo-qwen" \
#     --run_name "grpo-qwen1.5b-deepspeed"
