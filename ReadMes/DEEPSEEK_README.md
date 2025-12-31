# DeepSeek-R1-Distill-Qwen-7B Support

This document describes how to use DeepSeek-R1-Distill-Qwen-7B with the RLcausality codebase.

## Overview

DeepSeek-R1-Distill-Qwen-7B is a 7B parameter reasoning model based on Qwen2.5-Math-7B, fine-tuned using reasoning data from DeepSeek-R1. It's specifically designed for reasoning tasks, making it an excellent candidate for causal reasoning with GRPO training.

### Key Features
- **Model Size**: 7B parameters (larger than Qwen2.5-1.5B and Phi-3.5-mini-3.8B)
- **Architecture**: Based on Qwen2.5 (compatible with Qwen attention layers)
- **Context Length**: 32,768 tokens (we use 1024 for memory efficiency)
- **Precision**: BF16 recommended (FP16 for Turing GPUs)
- **Reasoning**: Pre-trained with `<think>` tags for chain-of-thought reasoning
- **Temperature**: 0.5-0.7 recommended (0.6 default)

### Differences from Other Models

| Feature | Qwen2.5-1.5B | Phi-3.5-mini | DeepSeek-R1-7B |
|---------|-------------|--------------|----------------|
| Parameters | 1.5B | 3.8B | 7B |
| Architecture | Qwen2.5 | Phi-3 | Qwen2.5-based |
| Reasoning Focus | No | Moderate | **Strong** |
| Chat Template | Optional | Required | Required |
| Memory (4-bit) | ~2GB | ~4GB | ~7-8GB |
| Training Time | Fast | Medium | Slower |
| Expected Performance | Baseline | +10% | +15-20% (est.) |

## Files Added

### Configuration Files
- `configs/sft_config_deepseek.yaml` - SFT training configuration
- `configs/grpo_config_deepseek.yaml` - GRPO training configuration

### Scripts
- `scripts/data_preprocessing_deepseek.py` - Data preprocessing with DeepSeek chat template
- `submit_preprocessing_deepseek.slurm` - SLURM job for preprocessing
- `submit_sft_deepseek.slurm` - SLURM job for SFT training
- `submit_grpo_deepseek.slurm` - SLURM job for GRPO training
- `run_deepseek_pipeline.sh` - Complete training pipeline
- `evaluate_all_models.sh` - Evaluate all three model families

## Quick Start

### Option 1: Direct GRPO Training (Fastest - Recommended for DeepSeek)

Train GRPO directly from the base model without SFT (DeepSeek-R1 is already pre-trained for reasoning):

```bash
# Just preprocessing + GRPO (skips SFT)
sbatch submit_preprocessing_deepseek.slurm
# Wait for preprocessing to complete, then:
sbatch submit_grpo_deepseek.slurm
```

**Why this works well**: DeepSeek-R1 is already pre-trained with reasoning capabilities, so it can do GRPO training directly without needing SFT first. This saves ~6-8 hours of training time!

### Option 2: Full Pipeline (SFT + GRPO)

Run the complete pipeline (preprocessing → SFT → GRPO) with job dependencies:

```bash
bash run_deepseek_pipeline.sh
```

This will submit three jobs that run sequentially:
1. Data preprocessing (1 hour)
2. SFT training (8 hours)
3. GRPO training (12 hours)

**Note**: To use this option, you need to change the config to use the SFT model. Edit `configs/grpo_config_deepseek.yaml`:
```yaml
# Change line 7 from:
model_name_or_path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# To:
model_name_or_path: "./models/deepseek_sft_model"
```

### Option 2: Step-by-Step

Run each step individually:

```bash
# Step 1: Preprocess data with DeepSeek chat template
sbatch submit_preprocessing_deepseek.slurm

# Step 2: Train SFT model (wait for preprocessing to complete)
sbatch submit_sft_deepseek.slurm

# Step 3: Train GRPO model (wait for SFT to complete)
sbatch submit_grpo_deepseek.slurm
```

### Option 3: Manual Execution

For debugging or customization:

```bash
# Activate environment
source venv/bin/activate

# Preprocessing
python scripts/data_preprocessing_deepseek.py --stage both

# SFT training (multi-GPU)
torchrun --nproc_per_node=2 \
    scripts/train_sft.py \
    configs/sft_config_deepseek.yaml

# GRPO training (multi-GPU)
torchrun --nproc_per_node=2 \
    scripts/train_grpo.py \
    configs/grpo_config_deepseek.yaml
```

## Evaluation

### Evaluate DeepSeek Models Only

```bash
# Evaluate SFT model
sbatch submit_evaluation.slurm ./models/deepseek_sft_model sft

# Evaluate GRPO model (latest checkpoint)
CHECKPOINT=$(ls -d ./models/deepseek_grpo_model/checkpoint-* | sort -V | tail -1)
sbatch submit_evaluation.slurm "${CHECKPOINT}" grpo
```

### Evaluate All Models (Qwen, Phi-3, DeepSeek)

```bash
bash evaluate_all_models.sh
```

## Configuration Details

### SFT Configuration ([configs/sft_config_deepseek.yaml](configs/sft_config_deepseek.yaml))

Key settings for 7B model:
- **Quantization**: 4-bit required for 11GB GPUs
- **LoRA rank**: 64 (higher for larger model)
- **Batch size**: 1 per device (conservative for memory)
- **Gradient accumulation**: 32 steps (effective batch size: 64)
- **Learning rate**: 2e-5 (lower for larger model)
- **Max sequence length**: 1024 tokens
- **Precision**: FP16 (or BF16 on Ampere+ GPUs)

### GRPO Configuration ([configs/grpo_config_deepseek.yaml](configs/grpo_config_deepseek.yaml))

Key settings for reasoning:
- **Max completion length**: 2048 tokens (for reasoning chains)
- **Temperature**: 0.6 (DeepSeek-R1 recommended)
- **Learning rate**: 5e-7 (very low for GRPO)
- **Num generations**: 4 samples per prompt
- **Batch size**: 1 per device
- **Gradient accumulation**: 16 steps

### Data Preprocessing

The preprocessing script uses DeepSeek-R1's chat template and encourages reasoning:

**SFT Format** (simple Yes/No):
```
<|user|>
Premise and Hypothesis:
[input text]

Is the hypothesis consistent with the premise? Answer with 'Yes' or 'No'.
<|assistant|>
Yes
```

**GRPO Format** (reasoning-focused):
```
<|user|>
You are an expert in causal reasoning. Given a premise describing statistical
relationships between variables, determine if the stated causal hypothesis is
consistent with the given information.

Think step by step using <think> tags to show your reasoning process. After
your reasoning, provide your final answer as either "Therefore: Yes" or
"Therefore: No".

Premise and Hypothesis:
[input text]

Begin your response with <think> and provide detailed step-by-step reasoning:
<|assistant|>
```

This format leverages DeepSeek-R1's pre-training with reasoning chains.

## Memory Requirements

DeepSeek-R1-Distill-Qwen-7B requires more memory than the smaller models:

| Configuration | Memory per GPU | Recommended Setup |
|---------------|----------------|-------------------|
| 4-bit + LoRA | ~7-8 GB | 2x RTX 2080 Ti (11GB) |
| 8-bit + LoRA | ~12-14 GB | 2x RTX 3090 (24GB) |
| Full FP16 | ~28-32 GB | 2x A100 (40GB) |

**Current setup (RTX 2080 Ti)**:
- 4-bit quantization: ✅ Fits
- Batch size 1: ✅ Required
- Gradient accumulation: ✅ 32 steps for effective batch size
- Gradient checkpointing: ✅ Essential

## Training Time Estimates

On 2x RTX 2080 Ti (11GB):

| Stage | Estimated Time | Output |
|-------|---------------|--------|
| Preprocessing | 30-60 minutes | `./data/processed/sft_deepseek/` <br> `./data/processed/grpo_deepseek/` |
| SFT Training | 6-8 hours | `./models/deepseek_sft_model/` |
| GRPO Training | 10-12 hours | `./models/deepseek_grpo_model/` |
| **Total** | **16-21 hours** | |

## Expected Performance

Based on model size and reasoning capabilities:

| Model | Expected Accuracy | Reasoning Quality |
|-------|------------------|-------------------|
| Qwen2.5-1.5B (SFT) | ~72% | Basic |
| Qwen2.5-1.5B (GRPO) | ~78% | Improved |
| Phi-3.5-mini (SFT) | ~75% | Good |
| Phi-3.5-mini (GRPO) | ~85% | Strong |
| **DeepSeek-R1-7B (SFT)** | **~78-80%** | **Strong** |
| **DeepSeek-R1-7B (GRPO)** | **~88-92%** | **Excellent** |

*Note: These are estimates. Actual performance may vary.*

## DeepSpeed Integration (For Memory Issues)

**All SLURM scripts now use DeepSpeed by default!**

If you're hitting memory issues, the training scripts automatically use:
- **SFT**: DeepSpeed ZeRO Stage 2 (fast, good memory savings)
- **GRPO**: DeepSpeed ZeRO Stage 3 (maximum memory savings)

This enables:
- ✅ Model sharding across GPUs (2×11GB ≈ 22GB effective memory)
- ✅ CPU offloading for optimizer states
- ✅ Batch size of 1 with GRPO (instead of 2)

**For details, see [DEEPSPEED_GUIDE.md](DEEPSPEED_GUIDE.md)**

### How DeepSpeed Helps

| Without DeepSpeed | With DeepSpeed ZeRO-3 |
|------------------|----------------------|
| ❌ 15GB per GPU → OOM | ✅ 7.5GB per GPU → Fits! |
| Must use batch_size=2 | Can use batch_size=1 |
| Runs out of memory | Trains successfully |

## Troubleshooting

### GRPO Batch Size Error

If you see: `ValueError: The global eval batch size (X) must be divisible by num_generations (4)`

**Solution**: The global batch size (per_device_batch_size * num_gpus) must be divisible by `num_generations`. For 2 GPUs:
- If `num_generations = 4`, then `per_device_eval_batch_size` must be at least 2
- Current config is already set correctly: `per_device_eval_batch_size: 2`

### Out of Memory (OOM) Errors

If you encounter OOM errors during training:

1. **Reduce batch size** (already at minimum of 1)
2. **Reduce max_seq_length**:
   ```yaml
   max_seq_length: 768  # or 512
   ```
3. **Use smaller LoRA rank**:
   ```yaml
   lora_r: 32  # instead of 64
   ```
4. **Reduce gradient accumulation** (trade-off: smaller effective batch size):
   ```yaml
   gradient_accumulation_steps: 16  # instead of 32
   ```

### Slow Training

If training is too slow:

1. **Enable Flash Attention** (if supported):
   ```yaml
   use_flash_attention: true
   ```
2. **Use more GPUs** (update SLURM `--gres=gpu:4`)
3. **Reduce dataset size**:
   ```yaml
   max_train_samples: 5000  # instead of 10000
   ```

### Model Not Loading

If the model fails to download:

```bash
# Pre-download the model
python -c "from transformers import AutoModelForCausalLM; \
           AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='./data/cache')"
```

## Comparison with Other Models

### When to Use DeepSeek-R1-7B

**Use DeepSeek-R1-7B when:**
- ✅ You need the best possible reasoning performance
- ✅ You have sufficient GPU memory (11GB+ per GPU)
- ✅ You can afford longer training times
- ✅ Your task benefits from chain-of-thought reasoning
- ✅ You're doing GRPO (reasoning-focused training)

**Use Qwen2.5-1.5B when:**
- ✅ You need fast iteration/debugging
- ✅ You have limited GPU memory
- ✅ You want a baseline model quickly
- ✅ Training time is critical

**Use Phi-3.5-mini when:**
- ✅ You want a good balance of size and performance
- ✅ You need stronger baseline than Qwen but smaller than DeepSeek
- ✅ You want the Phi-3 architecture specifically

## Model Architecture Notes

DeepSeek-R1-Distill-Qwen-7B uses the **Qwen2.5 architecture**:

- **Attention**: Multi-head attention with separate Q, K, V projections
- **LoRA targets**: `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj`
- **Activation**: SwiGLU (gated FFN)
- **Normalization**: RMSNorm
- **Positional encoding**: RoPE (Rotary Position Embedding)

This is the same architecture as Qwen2.5-1.5B, so the LoRA targets are identical.

## References

- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [Qwen2.5 Documentation](https://huggingface.co/Qwen)
- [GRPO Training Guide](https://huggingface.co/docs/trl/grpo_trainer)

## Support

For issues specific to DeepSeek support, check:
1. Model logs: `logs/sft_deepseek_*.out`, `logs/grpo_deepseek_*.out`
2. GPU memory: `nvidia-smi`
3. Job status: `squeue -u $USER`
4. Configuration files: `configs/sft_config_deepseek.yaml`, `configs/grpo_config_deepseek.yaml`

## License

DeepSeek-R1-Distill-Qwen-7B follows the DeepSeek license. Check the [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) for details.
