# DeepSpeed Integration Guide

This guide explains how to use DeepSpeed ZeRO optimization for memory-efficient training of large models like DeepSeek-R1-Distill-Qwen-7B.

## What is DeepSpeed?

**DeepSpeed** is a deep learning optimization library that enables:
- **ZeRO (Zero Redundancy Optimizer)**: Shards model states across GPUs
- **CPU Offloading**: Moves optimizer states and parameters to CPU RAM
- **Effective Memory Pooling**: 2×11GB GPUs ≈ 22GB effective VRAM

### ZeRO Stages

| Stage | What's Sharded | Memory Savings | Speed | Use Case |
|-------|---------------|----------------|-------|----------|
| **ZeRO-1** | Optimizer states | ~4x | Fast | Small models |
| **ZeRO-2** | Optimizer + Gradients | ~8x | Fast | Medium models (SFT) |
| **ZeRO-3** | Optimizer + Gradients + Parameters | ~15x+ | Slower | Large models (GRPO) |

## Why DeepSpeed for DeepSeek?

Without DeepSpeed, the 7B model with GRPO hits OOM errors because:
- **Model weights**: ~7GB (4-bit quantized)
- **Optimizer states**: ~3-4GB per GPU
- **Gradients**: ~2-3GB per GPU
- **Activations**: ~2-3GB per GPU with `num_generations=4`
- **Total**: ~14-17GB per GPU → **Exceeds 11GB limit**

With DeepSpeed ZeRO-3:
- Model weights sharded across 2 GPUs: 7GB → 3.5GB each
- Optimizer offloaded to CPU RAM
- Parameters offloaded to CPU RAM
- **Effective VRAM**: ~6-8GB per GPU ✅ Fits!

## Quick Start

### GRPO Training with DeepSpeed ZeRO-3 (Recommended)

```bash
# The SLURM script now uses DeepSpeed by default!
sbatch submit_grpo_deepseek.slurm
```

This will automatically use `configs/accelerate_deepspeed_zero3.yaml` for maximum memory savings.

### SFT Training with DeepSpeed ZeRO-2

```bash
# SFT also uses DeepSpeed (ZeRO-2 for better speed)
sbatch submit_sft_deepseek.slurm
```

## Configuration Files

### ZeRO-3 Config ([configs/accelerate_deepspeed_zero3.yaml](configs/accelerate_deepspeed_zero3.yaml))

Used for **GRPO** training (maximum memory savings):

```yaml
distributed_type: DEEPSPEED
num_processes: 2
mixed_precision: fp16

deepspeed_config:
  zero_stage: 3                      # Shard everything
  offload_optimizer_device: cpu      # Move optimizer to CPU
  offload_param_device: cpu          # Move parameters to CPU
  gradient_accumulation_steps: 32
  gradient_clipping: 1.0
  zero3_init_flag: true              # Required for ZeRO-3
  zero3_save_16bit_model: true       # Save weights in FP16
```

### ZeRO-2 Config ([configs/accelerate_deepspeed_zero2.yaml](configs/accelerate_deepspeed_zero2.yaml))

Used for **SFT** training (faster, less memory savings):

```yaml
distributed_type: DEEPSPEED
num_processes: 2
mixed_precision: fp16

deepspeed_config:
  zero_stage: 2                      # Shard optimizer + gradients only
  offload_optimizer_device: cpu      # Move optimizer to CPU
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
```

## How to Use

### Option 1: Use Updated SLURM Scripts (Recommended)

The SLURM scripts are already updated to use DeepSpeed:

```bash
# GRPO with ZeRO-3
sbatch submit_grpo_deepseek.slurm

# SFT with ZeRO-2
sbatch submit_sft_deepseek.slurm
```

### Option 2: Manual Launch

If running manually:

```bash
# Activate environment
source venv/bin/activate

# GRPO with DeepSpeed ZeRO-3
accelerate launch \
    --config_file configs/accelerate_deepspeed_zero3.yaml \
    scripts/train_grpo.py \
    configs/grpo_config_deepseek.yaml

# SFT with DeepSpeed ZeRO-2
accelerate launch \
    --config_file configs/accelerate_deepspeed_zero2.yaml \
    scripts/train_sft.py \
    configs/sft_config_deepseek.yaml
```

## Configuration Changes for DeepSpeed

### GRPO Config Updated

The batch size in `configs/grpo_config_deepseek.yaml` has been optimized:

```yaml
# OLD (without DeepSpeed):
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 16

# NEW (with DeepSpeed):
per_device_train_batch_size: 1     # ZeRO-3 allows batch_size=1
per_device_eval_batch_size: 2      # Still 2 for GRPO constraint
gradient_accumulation_steps: 32    # Doubled to maintain effective batch size
```

**Why eval batch is still 2?**
- GRPO requires: `global_eval_batch_size % num_generations == 0`
- With 2 GPUs: `2 devices × 2 = 4` → divisible by `num_generations=4` ✅

## Memory Comparison

### Without DeepSpeed (DDP)

| Component | Per GPU | 2 GPUs Total |
|-----------|---------|--------------|
| Model (4-bit) | 7 GB | 14 GB (replicated) |
| Optimizer | 3 GB | 6 GB (replicated) |
| Gradients | 2 GB | 4 GB (replicated) |
| Activations | 3 GB | 6 GB |
| **Total** | **15 GB** | **30 GB** |
| **Result** | ❌ **OOM** | ❌ **Exceeds 22GB** |

### With DeepSpeed ZeRO-3

| Component | Per GPU | 2 GPUs Total |
|-----------|---------|--------------|
| Model (sharded) | 3.5 GB | 7 GB (split) |
| Optimizer (CPU) | 0 GB | 0 GB (offloaded) |
| Gradients (sharded) | 1 GB | 2 GB (split) |
| Activations | 3 GB | 6 GB |
| **Total GPU** | **7.5 GB** | **15 GB** |
| **Total CPU** | 3 GB | 6 GB (optimizer in RAM) |
| **Result** | ✅ **Fits!** | ✅ **Under 22GB** |

## Performance Impact

DeepSpeed adds some overhead due to CPU↔GPU communication:

| Approach | Speed | Memory | Recommended For |
|----------|-------|--------|-----------------|
| **DDP (torchrun)** | Fastest | High | Small models that fit |
| **ZeRO-2** | ~90% speed | Medium | SFT training |
| **ZeRO-3** | ~70-80% speed | Lowest | GRPO training, large models |

**For DeepSeek-R1-7B**:
- **SFT**: ZeRO-2 (good balance)
- **GRPO**: ZeRO-3 (required for memory)

## Troubleshooting

### Error: "Duplicate GPU detected: rank 0 and rank 1 both on CUDA device XXX"

**Problem**: Both DeepSpeed ranks are trying to use the same GPU instead of separate GPUs.

**Cause**: SLURM sets `CUDA_VISIBLE_DEVICES` with full GPU IDs (e.g., "21000,21001"), but DeepSpeed expects simple indices like "0,1".

**Solution**: The SLURM scripts now automatically fix this by **unsetting** `CUDA_VISIBLE_DEVICES` to let DeepSpeed auto-detect GPUs.

The updated scripts include:
```bash
# Unset SLURM's CUDA_VISIBLE_DEVICES and let DeepSpeed handle it
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Original CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    unset CUDA_VISIBLE_DEVICES
    echo "Unset CUDA_VISIBLE_DEVICES to let DeepSpeed auto-detect GPUs"
fi
```

Or run manually without CUDA_VISIBLE_DEVICES:
```bash
# Make sure CUDA_VISIBLE_DEVICES is not set
unset CUDA_VISIBLE_DEVICES

# Then launch
accelerate launch \
    --config_file configs/accelerate_deepspeed_zero3.yaml \
    scripts/train_grpo.py \
    configs/grpo_config_deepseek.yaml
```

**Why this works**: SLURM sets CUDA_VISIBLE_DEVICES with full GPU IDs (like "21000,21001"), but DeepSpeed expects simple indices. By unsetting it, DeepSpeed will use `--gres=gpu:2` from SLURM to correctly detect and assign 2 GPUs.

### Error: "DeepSpeed is not installed"

```bash
pip install deepspeed
```

Or if already installed:
```bash
pip install --upgrade deepspeed
```

### Error: "accelerate command not found"

```bash
pip install accelerate
```

### Still Getting OOM Errors

If ZeRO-3 still gives OOM:

1. **Reduce num_generations** (less activation memory):
   ```yaml
   num_generations: 2  # instead of 4
   ```

2. **Enable more CPU offloading** - Edit `configs/accelerate_deepspeed_zero3.yaml`:
   ```yaml
   deepspeed_config:
     offload_param_device: cpu
     offload_optimizer_device: cpu
     stage3_max_live_parameters: 5e8  # Reduce to 500M
   ```

3. **Reduce max_completion_length**:
   ```yaml
   max_completion_length: 1024  # instead of 2048
   ```

### DeepSpeed Process Hangs

If training hangs at initialization:

1. **Check NCCL settings**:
   ```bash
   export NCCL_DEBUG=INFO
   ```

2. **Verify GPU communication**:
   ```bash
   nvidia-smi topo -m
   ```

3. **Use different backend** - Edit accelerate config:
   ```yaml
   distributed_type: DEEPSPEED
   deepspeed_multinode_launcher: pdsh  # or 'standard'
   ```

### Checkpoint Loading Issues

When loading ZeRO-3 checkpoints, make sure to use the same DeepSpeed config:

```bash
accelerate launch \
    --config_file configs/accelerate_deepspeed_zero3.yaml \
    scripts/evaluate_model.py \
    --model_path ./models/deepseek_grpo_model/checkpoint-100
```

## Advanced: Custom DeepSpeed Config

If you need to customize DeepSpeed further:

```bash
# Generate default config
accelerate config

# Or edit the YAML files directly
vim configs/accelerate_deepspeed_zero3.yaml
```

Key parameters to adjust:

```yaml
deepspeed_config:
  # Memory vs Speed tradeoff
  stage3_max_live_parameters: 1e9        # Lower = less GPU memory, slower
  stage3_max_reuse_distance: 1e9         # Lower = less GPU memory, slower

  # Offloading settings
  offload_optimizer_device: cpu          # 'none' for faster, 'cpu' for more memory
  offload_param_device: cpu              # 'none' for faster, 'cpu' for more memory

  # Communication optimization
  overlap_comm: true                     # Overlap computation with communication
  reduce_bucket_size: 5e8                # Smaller = less memory, more communication
```

## Comparison with Other Approaches

### 1. Standard DDP (torchrun)
- ✅ Fastest
- ❌ Highest memory (replicates model on each GPU)
- ❌ **Doesn't work for DeepSeek-R1-7B GRPO**

### 2. FSDP (Fully Sharded Data Parallel)
- ✅ Fast
- ✅ Good memory savings
- ⚠️ PyTorch 2.0+ required
- ⚠️ Less tested with TRL

### 3. DeepSpeed ZeRO (Current)
- ✅ Best memory efficiency
- ✅ Well-tested with TRL
- ✅ Works with older PyTorch
- ⚠️ Slightly slower than FSDP

## Summary

**For DeepSeek-R1-Distill-Qwen-7B training:**

| Task | Command | DeepSpeed Config | Why |
|------|---------|-----------------|-----|
| **Preprocessing** | `sbatch submit_preprocessing_deepseek.slurm` | None | CPU-only task |
| **SFT** | `sbatch submit_sft_deepseek.slurm` | ZeRO-2 | Good speed/memory balance |
| **GRPO** | `sbatch submit_grpo_deepseek.slurm` | ZeRO-3 | Required for memory |

**All SLURM scripts are already configured!** Just run them as usual.

## Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [Accelerate DeepSpeed Guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [TRL with DeepSpeed](https://huggingface.co/docs/trl/main/en/use_deepspeed)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)