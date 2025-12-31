# Qwen2.5-1.5B Configuration Guide

This project has been adapted to use **Qwen/Qwen2.5-1.5B** instead of Llama-2-7b. This smaller model offers:

- **Faster training** (~2-3x faster than 7B models)
- **Lower memory requirements** (can run on 2 GPUs instead of 4)
- **Shorter queue times** on HPC clusters
- **Still competitive performance** on reasoning tasks

## What Changed

### 1. Model Configuration

**Model**: `Qwen/Qwen2.5-1.5B` (1.5B parameters vs 7B)

**Files updated:**
- [configs/sft_config.yaml](configs/sft_config.yaml#L5)
- [scripts/train_sft.py](scripts/train_sft.py#L28)

### 2. Training Hyperparameters

#### SFT Training ([sft_config.yaml](configs/sft_config.yaml))

| Parameter | Qwen2.5-1.5B | Llama-2-7b | Reason |
|-----------|--------------|------------|--------|
| `lora_r` | 32 | 64 | Smaller model needs fewer LoRA parameters |
| `per_device_train_batch_size` | 8 | 4 | Can fit larger batches in memory |
| `gradient_accumulation_steps` | 2 | 4 | Adjusted for similar effective batch size |
| `learning_rate` | 3e-5 | 2e-5 | Slightly higher LR for smaller model |

**Effective batch size** (2 GPUs): `8 × 2 × 2 = 32`

#### GRPO Training ([grpo_config.yaml](configs/grpo_config.yaml))

| Parameter | Qwen2.5-1.5B | Llama-2-7b | Reason |
|-----------|--------------|------------|--------|
| `lora_r` | 16 | 32 | Smaller rank for additional adapters |
| `per_device_train_batch_size` | 4 | 2 | Can handle larger batches |
| `gradient_accumulation_steps` | 4 | 8 | Adjusted for batch size |
| `learning_rate` | 1e-6 | 5e-7 | Slightly higher for smaller model |

**Effective batch size** (2 GPUs): `4 × 4 × 2 = 32`

### 3. GPU Requirements

#### SLURM Configuration

**Before (Llama-2-7b):**
```bash
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00  # SFT
#SBATCH --time=48:00:00  # GRPO
```

**After (Qwen2.5-1.5B):**
```bash
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00  # SFT
#SBATCH --time=24:00:00  # GRPO
```

**Benefits:**
- **50% fewer GPUs** - easier to get allocated
- **50% less memory** - more efficient
- **50% less time** - faster training

## Memory Footprint Comparison

| Configuration | Llama-2-7b | Qwen2.5-1.5B | Savings |
|---------------|------------|--------------|---------|
| Model (bf16) | ~14 GB | ~3 GB | 78% |
| Model (4-bit) | ~3.5 GB | ~0.75 GB | 79% |
| + LoRA + Optimizer | ~6-8 GB | ~2-3 GB | 60-70% |
| **Total per GPU** | **~10-12 GB** | **~3-5 GB** | **~60%** |

With Qwen2.5-1.5B, you can:
- Train on consumer GPUs (RTX 3090/4090)
- Use fewer cluster GPUs
- Run larger batch sizes

## Training Time Estimates

On **2x A100 GPUs** (40GB):

| Stage | Dataset Size | Qwen2.5-1.5B | Llama-2-7b |
|-------|--------------|--------------|------------|
| **SFT** | 10k samples (default) | 1-2 hours | 3-4 hours |
| **SFT** | Full 206k samples | 4-6 hours | 12-16 hours |
| **GRPO** | 1 epoch | 8-12 hours | 24-30 hours |

**Note**: By default, SFT is configured to train on **10,000 samples** for faster experimentation. To use the full dataset, see configuration below.

## Adjusting Training Dataset Size

By default, SFT training uses **10,000 samples** for faster experimentation. You can change this in [configs/sft_config.yaml](configs/sft_config.yaml):

```yaml
# Use 10k samples (default - fast)
max_train_samples: 10000

# Use 50k samples (medium)
max_train_samples: 50000

# Use full dataset (206k samples - slow)
max_train_samples: null
```

Or override via command line:
```bash
# Train on 5k samples
torchrun --nproc_per_node=2 scripts/train_sft.py configs/sft_config.yaml --max_train_samples 5000

# Use full dataset
torchrun --nproc_per_node=2 scripts/train_sft.py configs/sft_config.yaml --max_train_samples null
```

## Quick Start

### 1. Preprocess Data

```bash
python scripts/data_preprocessing.py
```

### 2. Run SFT Training

**Local/Interactive:**
```bash
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/train_sft.py configs/sft_config.yaml
```

**HPC (SLURM):**
```bash
sbatch submit_sft.slurm
```

### 3. Run GRPO Training

**Local/Interactive:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/train_grpo.py configs/grpo_config.yaml
```

**HPC (SLURM):**
```bash
sbatch submit_grpo.slurm
```

## Model-Specific Notes

### Qwen2.5 Architecture

Qwen2.5-1.5B uses:
- **Architecture**: Similar to Llama (decoder-only transformer)
- **Vocab size**: 151,936 tokens (larger than Llama's ~32k)
- **Context length**: 32,768 tokens (vs Llama-2's 4,096)
- **Target modules**: Same as Llama (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)

### Tokenizer

Qwen models use a **BPE tokenizer** with:
- Built-in `pad_token` (usually `<|endoftext|>`)
- `trust_remote_code=True` required for loading

### Performance Expectations

Based on similar reasoning benchmarks:
- **Qwen2.5-1.5B**: Competitive with Llama-2-7b on many tasks
- **After SFT**: Should achieve 70-80% accuracy on corr2cause
- **After GRPO**: Expected 75-85% with better reasoning quality

## Switching Back to Llama or Using Larger Models

To use a different model, update:

1. **[configs/sft_config.yaml](configs/sft_config.yaml#L5)**:
   ```yaml
   model_name_or_path: "meta-llama/Llama-2-7b-hf"  # or "Qwen/Qwen2.5-7B"
   ```

2. **Adjust hyperparameters** based on model size:
   - Larger models: smaller batch sizes, lower LR, more GPUs
   - Smaller models: larger batch sizes, higher LR, fewer GPUs

3. **Update SLURM scripts**:
   ```bash
   #SBATCH --gres=gpu:4  # For 7B models
   ```

## Recommended Models by Size

| Model | Parameters | GPUs | Use Case |
|-------|------------|------|----------|
| `Qwen/Qwen2.5-0.5B` | 0.5B | 1 | Testing, experimentation |
| `Qwen/Qwen2.5-1.5B` | 1.5B | 2 | **Current setup**, fast training |
| `Qwen/Qwen2.5-3B` | 3B | 2 | Better performance, still fast |
| `Qwen/Qwen2.5-7B` | 7B | 4 | High performance |
| `Qwen/Qwen2.5-14B` | 14B | 8 | Best performance (requires more resources) |

## Troubleshooting

### Model Download Issues

If `Qwen/Qwen2.5-1.5B` download fails:
```bash
# Pre-download the model
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./models/Qwen2.5-1.5B

# Update config to use local path
model_name_or_path: "./models/Qwen2.5-1.5B"
```

### Trust Remote Code Warning

Qwen models require `trust_remote_code=True`. This is already set in the code, but if you see warnings, this is expected and safe for official Qwen models.

### Out of Memory

If you still run out of memory:
1. Enable 4-bit quantization:
   ```yaml
   use_4bit: true
   ```
2. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 4
   ```
3. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 4
   ```

## Additional Resources

- **Qwen2.5 Model Card**: https://huggingface.co/Qwen/Qwen2.5-1.5B
- **Qwen2.5 Blog**: https://qwenlm.github.io/
- **TRL Documentation**: https://huggingface.co/docs/trl/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
