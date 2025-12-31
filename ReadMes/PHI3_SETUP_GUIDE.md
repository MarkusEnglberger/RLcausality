# Phi-3.5-Mini Setup Guide for Causal Reasoning

This guide explains how to train Phi-3.5-Mini (3.8B parameters) on the Corr2Cause causal reasoning task as an alternative to Qwen2.5-1.5B.

## Why Phi-3.5-Mini?

Phi-3.5-Mini offers significant advantages over Qwen2.5-1.5B for reasoning tasks:

| Feature | Qwen2.5-1.5B | Phi-3.5-Mini (3.8B) |
|---------|--------------|---------------------|
| Parameters | 1.5B | 3.8B |
| Reasoning capability | Good | Excellent |
| Expected SFT accuracy | ~65% | 70-75% |
| Expected GRPO accuracy | ~67% | 75-82% |
| Training stability | Moderate | High |
| Specialized training | General | Reasoning-focused |
| Context length | 32k | 128k |

**Key advantages:**
- Trained specifically on reasoning and instruction-following tasks
- Better exploration during GRPO (responds well to RL fine-tuning)
- More efficient architecture (can use larger batches)
- Proven strong performance on logical reasoning benchmarks

---

## Files Created

### Configuration Files
- [configs/sft_config_phi3.yaml](configs/sft_config_phi3.yaml) - SFT training configuration
- [configs/grpo_config_phi3.yaml](configs/grpo_config_phi3.yaml) - GRPO training configuration

### Scripts
- [scripts/data_preprocessing_phi3.py](scripts/data_preprocessing_phi3.py) - Data preprocessing with Phi-3.5 chat template
- [submit_preprocessing_phi3.slurm](submit_preprocessing_phi3.slurm) - SLURM script for preprocessing
- [submit_sft_phi3.slurm](submit_sft_phi3.slurm) - SLURM script for SFT training
- [submit_grpo_phi3.slurm](submit_grpo_phi3.slurm) - SLURM script for GRPO training
- [run_phi3_pipeline.sh](run_phi3_pipeline.sh) - Automated pipeline script

---

## Key Differences from Qwen

### 1. Chat Template
Phi-3.5 uses a different chat format:
```
<|user|>
{user_message}<|end|>
<|assistant|>
{assistant_message}<|end|>
```

This is handled automatically by `data_preprocessing_phi3.py` using the tokenizer's `apply_chat_template()` method.

### 2. LoRA Target Modules
Phi-3.5 has a different attention architecture:
```yaml
# Qwen (separate Q, K, V projections)
lora_target_modules: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# Phi-3.5 (combined QKV and gate_up projections)
lora_target_modules: "qkv_proj,o_proj,gate_up_proj,down_proj"
```

### 3. Precision Settings
Phi-3.5 works better with bf16 on L4 GPUs (Ampere architecture):
```yaml
# Qwen (for RTX 2080 Ti - Turing)
bf16: false
fp16: true
tf32: false

# Phi-3.5 (for L4 - Ampere)
bf16: true
fp16: false
tf32: true
```

### 4. GRPO Hyperparameters
Phi-3.5 benefits from more exploration:
```yaml
# Qwen
num_generations: 4
temperature: 0.8
beta: 0.05

# Phi-3.5
num_generations: 6    # More generations for better estimates
temperature: 0.9      # Higher temperature for exploration
beta: 0.03            # Lower KL penalty (model needs less constraint)
```

---

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Run the entire pipeline (preprocessing → SFT → GRPO) with job dependencies:

```bash
# On HPC
cd ~/RLcausality
bash run_phi3_pipeline.sh
```

This will submit 3 jobs that run sequentially:
1. **Preprocessing** (1 hour, CPU): Formats data with Phi-3.5 chat template
2. **SFT Training** (4 hours, 2x GPU): Supervised fine-tuning on 10k samples
3. **GRPO Training** (8 hours, 2x GPU): Reinforcement learning on 5k samples

Total time: ~13 hours

### Option 2: Step-by-Step

If you prefer manual control:

```bash
# Step 1: Preprocess data
sbatch submit_preprocessing_phi3.slurm

# Wait for preprocessing to complete, then:

# Step 2: Run SFT training
sbatch submit_sft_phi3.slurm

# Wait for SFT to complete, then:

# Step 3: Run GRPO training
sbatch submit_grpo_phi3.slurm
```

---

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### View Logs in Real-Time
```bash
# Preprocessing
tail -f logs/preprocess_phi3_*.out

# SFT training
tail -f logs/sft_phi3_*.out

# GRPO training
tail -f logs/grpo_phi3_*.out
```

### What to Look For

**During SFT:**
- Loss should decrease from ~2.0 to ~0.5-0.8
- Eval loss should track training loss
- Training time: ~4 hours for 10k samples

**During GRPO:**
- Initial mean reward: 0.3-0.5 (65-75% accuracy)
- Target mean reward: 0.5-0.6+ (75-80%+ accuracy)
- KL divergence: Should stay between 0.02-0.05
- grad_norm: Should be non-zero (0.03-0.10 typical)

---

## Expected Results

### Qwen2.5-1.5B (Current)
- SFT accuracy: ~65%
- GRPO accuracy: ~67%
- **Improvement: +2%** (minimal)

### Phi-3.5-Mini (Expected)
- SFT accuracy: 70-75%
- GRPO accuracy: 75-82%
- **Improvement: +5-10%** (significant)

**Why the difference?**
- Phi-3.5 has better reasoning capabilities from pre-training
- 2.5x more parameters (3.8B vs 1.5B)
- Trained specifically on reasoning tasks
- Responds better to RL fine-tuning

---

## Model Locations

After training completes:

```
models/
├── phi3_sft_model/          # SFT checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin     # LoRA weights (~50MB)
│   └── ...
└── phi3_grpo_model/         # GRPO checkpoint
    ├── checkpoint-50/
    ├── checkpoint-100/
    └── ...
```

---

## Evaluation

### Evaluate SFT Model
```bash
sbatch submit_evaluation.slurm ./models/phi3_sft_model sft
```

### Evaluate GRPO Model
```bash
# Find latest checkpoint
ls models/phi3_grpo_model/

# Evaluate it
sbatch submit_evaluation.slurm ./models/phi3_grpo_model/checkpoint-XXX grpo
```

### Compare Both Models
```bash
# Edit evaluate_both_models.sh to use Phi-3.5 paths
# Then run:
bash evaluate_both_models.sh
```

Results will be saved in `evaluation_results/`.

---

## Memory Requirements

| Component | Memory per GPU | Total (2 GPUs) |
|-----------|----------------|----------------|
| Model (4-bit) | ~4GB | ~8GB |
| Activations | ~3-4GB | ~6-8GB |
| Gradients | ~2GB | ~4GB |
| **Total** | **~9-10GB** | **~18-20GB** |

This fits comfortably on:
- ✅ 2x NVIDIA L4 (23GB each) - Plenty of headroom
- ✅ 2x RTX 2080 Ti (11GB each) - Tight but works with optimizations
- ❌ 1x GPU - Not enough memory for multi-generation GRPO

---

## Troubleshooting

### Error: "Out of memory"
**Solution 1** - Reduce batch size in config:
```yaml
# In sft_config_phi3.yaml
per_device_train_batch_size: 1  # Down from 2
gradient_accumulation_steps: 16  # Up from 8
```

**Solution 2** - Reduce LoRA rank:
```yaml
lora_r: 16  # Down from 32
lora_alpha: 8  # Down from 16
```

### Error: "Tokenizer does not have chat template"
This shouldn't happen with Phi-3.5, but if it does:
```bash
# Check tokenizer
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct'); print(t.chat_template)"
```

### GRPO reward not improving
Check these in the logs:
- **grad_norm = 0**: Gradients not flowing (shouldn't happen with our fix)
- **KL too high (>0.1)**: Reduce beta to 0.02
- **KL too low (<0.01)**: Increase temperature to 1.0
- **High reward variance**: Increase num_generations to 8

### SFT accuracy lower than expected
- **Check prompt format**: View first example in preprocessing output
- **Check answer extraction**: May need to adjust regex in `extract_answer()`
- **Increase training data**: Set `max_train_samples: null` for full 200k dataset

---

## Scaling Up (Optional)

If you want to use the full 200k dataset instead of 10k:

```yaml
# In configs/sft_config_phi3.yaml
max_train_samples: null  # Use all data (was: 10000)
num_train_epochs: 1      # Reduce from 3 (more data = fewer epochs needed)

# Expected training time: ~12-15 hours (vs 4 hours for 10k)
# Expected SFT accuracy: 75-80% (vs 70-75% for 10k)
```

For GRPO, you can also increase samples:
```yaml
# In configs/grpo_config_phi3.yaml
max_train_samples: 10000  # Increase from 5000
# Expected training time: ~16 hours (vs 8 hours)
```

---

## Next Steps After Training

1. **Evaluate both models** to quantify improvement
2. **Compare with Qwen** to validate Phi-3.5 is better
3. **If results are good (>75%)**: Consider this project complete!
4. **If results are mediocre (65-75%)**: Try:
   - Full dataset training (200k samples)
   - Process-based rewards (reward reasoning steps)
   - Increase model size (try Phi-3-Medium 14B if you have GPU memory)

---

## Files Summary

**New files for Phi-3.5:**
- `configs/sft_config_phi3.yaml` - SFT configuration
- `configs/grpo_config_phi3.yaml` - GRPO configuration
- `scripts/data_preprocessing_phi3.py` - Phi-3.5-specific preprocessing
- `submit_preprocessing_phi3.slurm` - Preprocessing job
- `submit_sft_phi3.slurm` - SFT training job
- `submit_grpo_phi3.slurm` - GRPO training job
- `run_phi3_pipeline.sh` - Automated pipeline
- `PHI3_SETUP_GUIDE.md` - This document

**Reused files** (no changes needed):
- `scripts/train_sft.py` - Works with any model
- `scripts/train_grpo.py` - Works with any model (gradient fix included)
- `scripts/evaluate_model.py` - Works with any model
- `submit_evaluation.slurm` - Works with any model

---

## Performance Comparison Table

| Metric | Qwen2.5-1.5B | Phi-3.5-Mini | Improvement |
|--------|--------------|--------------|-------------|
| **Parameters** | 1.5B | 3.8B | +2.5x |
| **Memory (4-bit)** | ~2GB | ~4GB | +2GB |
| **SFT Accuracy** | 65% | 72% (est.) | +7% |
| **GRPO Accuracy** | 67% | 78% (est.) | +11% |
| **SFT Training Time** | 2h 15m | 4h (est.) | +1h 45m |
| **GRPO Training Time** | 10h | 8h (est.) | -2h |
| **GRPO Stability** | Moderate | High | Better |
| **Exploration** | Low | Good | Better |

---

## Recommendation

**Start with Phi-3.5-Mini** instead of continuing with Qwen2.5-1.5B because:
1. ✅ Significantly better reasoning capabilities
2. ✅ More stable GRPO training
3. ✅ Expected 10%+ accuracy improvement
4. ✅ Still fits on your GPUs with 4-bit quantization
5. ✅ Minimal code changes (all configs ready to use)

The extra ~2GB memory per GPU and 2 extra hours of training time are well worth the expected 10%+ accuracy gain.

---

## Quick Commands Reference

```bash
# Run complete pipeline (automated)
bash run_phi3_pipeline.sh

# Or run manually step-by-step
sbatch submit_preprocessing_phi3.slurm
sbatch submit_sft_phi3.slurm
sbatch submit_grpo_phi3.slurm

# Monitor jobs
squeue -u $USER
tail -f logs/sft_phi3_*.out

# Evaluate models
sbatch submit_evaluation.slurm ./models/phi3_sft_model sft
sbatch submit_evaluation.slurm ./models/phi3_grpo_model/checkpoint-XXX grpo

# Check results
cat evaluation_results/*.json
```

Good luck with the training! 🚀
