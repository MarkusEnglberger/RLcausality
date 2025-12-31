# Direct GRPO Training Guide (No SFT Required)

This guide explains how to train GRPO models directly from base models, skipping the SFT stage entirely.

## Why Direct GRPO Works

**DeepSeek-R1-Distill-Qwen-7B** is already pre-trained with reasoning capabilities, so it can be trained with GRPO directly without needing task-specific SFT first. This:
- **Saves 6-8 hours** of SFT training time
- **Reduces total training to ~10-12 hours** (just preprocessing + GRPO)
- Still achieves excellent performance due to pre-existing reasoning abilities

## Quick Start: Direct GRPO for DeepSeek

```bash
# Step 1: Preprocess data
sbatch submit_preprocessing_deepseek.slurm

# Step 2: Train GRPO directly (no SFT needed!)
sbatch submit_grpo_deepseek.slurm
```

**That's it!** The model will train GRPO directly from `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.

## How It Works

The config file [configs/grpo_config_deepseek.yaml](configs/grpo_config_deepseek.yaml) is set to use the base model:

```yaml
# Line 7: Use base model directly
model_name_or_path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Quantization is enabled for memory efficiency
use_4bit: true

# Batch sizes are set to satisfy GRPO constraints
per_device_train_batch_size: 2  # Global batch = 2 * 2 GPUs = 4
per_device_eval_batch_size: 2   # Must be divisible by num_generations (4)
```

## Switching Between Direct GRPO and SFT→GRPO

### To Use Direct GRPO (Default)
```yaml
# configs/grpo_config_deepseek.yaml line 7:
model_name_or_path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### To Use SFT→GRPO Pipeline
```yaml
# configs/grpo_config_deepseek.yaml line 7:
model_name_or_path: "./models/deepseek_sft_model"
```

Then uncomment line 6 and comment line 7 if you want to see both options.

## Other Models

### Can Qwen2.5-1.5B Do Direct GRPO?

**Yes, but not recommended**. Qwen2.5-1.5B is NOT pre-trained for reasoning, so direct GRPO may not converge well.

To try it, edit [configs/grpo_config.yaml](configs/grpo_config.yaml):
```yaml
# Change line 5 from:
model_name_or_path: "./models/sft_model"
# To:
model_name_or_path: "Qwen/Qwen2.5-1.5B"

# Add quantization:
use_4bit: true
use_8bit: false
```

### Can Phi-3.5-mini Do Direct GRPO?

**Maybe**. Phi-3.5-mini-instruct is instruction-tuned, so it might work reasonably well.

To try it, edit [configs/grpo_config_phi3.yaml](configs/grpo_config_phi3.yaml):
```yaml
# Change line 6 from:
model_name_or_path: "./models/phi3_sft_model/checkpoint-900"
# To:
model_name_or_path: "microsoft/Phi-3.5-mini-instruct"

# Add quantization:
use_4bit: true
use_8bit: false
```

## Training Time Comparison

### DeepSeek-R1-Distill-Qwen-7B

| Approach | Time | Steps |
|----------|------|-------|
| **Direct GRPO** | **~10-12 hours** | Preprocessing (1h) → GRPO (10-12h) |
| SFT→GRPO | ~16-21 hours | Preprocessing (1h) → SFT (6-8h) → GRPO (10-12h) |

**Savings: 6-8 hours** by skipping SFT!

### Qwen2.5-1.5B

| Approach | Time | Steps |
|----------|------|-------|
| Direct GRPO | ~6-8 hours | Preprocessing (30m) → GRPO (6-8h) |
| **SFT→GRPO** | **~9-12 hours** | Preprocessing (30m) → SFT (3-4h) → GRPO (6-8h) |

**Recommendation: Use SFT→GRPO** for Qwen (better convergence).

### Phi-3.5-mini

| Approach | Time | Steps |
|----------|------|-------|
| Direct GRPO | ~8-10 hours | Preprocessing (45m) → GRPO (8-10h) |
| **SFT→GRPO** | **~13-16 hours** | Preprocessing (45m) → SFT (5-6h) → GRPO (8-10h) |

**Recommendation: Try Direct GRPO first** (instruction-tuned baseline).

## Expected Performance

### DeepSeek-R1-7B

| Approach | Expected Accuracy | Notes |
|----------|------------------|-------|
| **Direct GRPO** | **~85-90%** | ✅ Recommended - saves time, good performance |
| SFT→GRPO | ~88-92% | Slightly better, but takes longer |

### Qwen2.5-1.5B

| Approach | Expected Accuracy | Notes |
|----------|------------------|-------|
| Direct GRPO | ~72-75% | ❌ Poor - model not pre-trained for reasoning |
| **SFT→GRPO** | **~78%** | ✅ Recommended - much better convergence |

### Phi-3.5-mini

| Approach | Expected Accuracy | Notes |
|----------|------------------|-------|
| **Direct GRPO** | **~80-83%** | ✅ Worth trying - instruction-tuned baseline |
| SFT→GRPO | ~85% | Better, but only ~2-5% improvement |

## Important: GRPO Batch Size Constraint

GRPO requires: `global_batch_size % num_generations == 0`

With 2 GPUs and `num_generations: 4`:
- `per_device_batch_size: 1` → global batch = 2 → ❌ **FAILS** (2 not divisible by 4)
- `per_device_batch_size: 2` → global batch = 4 → ✅ **WORKS** (4 divisible by 4)

**Current configs are already fixed** with `per_device_train_batch_size: 2`.

## Troubleshooting

### Error: "global eval batch size must be divisible by num_generations"

**Solution**: Increase `per_device_eval_batch_size` in your GRPO config:
```yaml
per_device_eval_batch_size: 2  # For 2 GPUs with num_generations=4
```

### Out of Memory with Batch Size 2

If you get OOM with `per_device_batch_size: 2`, you have two options:

**Option 1**: Reduce `num_generations` (trade-off: fewer samples for GRPO)
```yaml
num_generations: 2  # Now batch_size=1 works (1*2=2, divisible by 2)
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
```

**Option 2**: Use gradient accumulation (keeps effective batch size)
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 32  # Double from 16 to compensate
```

### Model Not Loading

Make sure the model name is correct in the config:
- DeepSeek: `"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"`
- Qwen: `"Qwen/Qwen2.5-1.5B"`
- Phi-3: `"microsoft/Phi-3.5-mini-instruct"`

## Summary: When to Use Direct GRPO

| Model | Direct GRPO? | Reason |
|-------|-------------|--------|
| **DeepSeek-R1-7B** | ✅ **Yes (Recommended)** | Pre-trained for reasoning, saves 6-8 hours |
| Qwen2.5-1.5B | ❌ No | Not pre-trained for reasoning, poor convergence |
| Phi-3.5-mini | ⚠️ Maybe | Instruction-tuned, worth trying but SFT→GRPO is safer |

## Commands Summary

```bash
# DeepSeek: Direct GRPO (Recommended)
sbatch submit_preprocessing_deepseek.slurm
sbatch submit_grpo_deepseek.slurm

# Qwen: SFT→GRPO (Recommended)
sbatch submit_preprocessing.slurm
sbatch submit_sft.slurm
sbatch submit_grpo.slurm

# Phi-3: Try Direct GRPO first
sbatch submit_preprocessing_phi3.slurm
sbatch submit_grpo_phi3.slurm  # After editing config for direct mode
```

## Related Documentation

- [DEEPSEEK_README.md](DEEPSEEK_README.md) - Full DeepSeek guide
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md) - Compare all three models
- [configs/grpo_config_deepseek.yaml](configs/grpo_config_deepseek.yaml) - DeepSeek GRPO config
