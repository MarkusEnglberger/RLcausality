# Model Comparison: Qwen2.5-1.5B vs Phi-3.5-mini vs DeepSeek-R1-7B

This document compares the three supported models in the RLcausality codebase.

## Quick Reference Table

| Feature | Qwen2.5-1.5B | Phi-3.5-mini-instruct | DeepSeek-R1-Distill-Qwen-7B |
|---------|-------------|----------------------|----------------------------|
| **Parameters** | 1.5B | 3.8B | 7B |
| **Architecture** | Qwen2.5 | Phi-3 | Qwen2.5-based |
| **Context Length** | 32K tokens | 128K tokens | 32K tokens |
| **Memory (4-bit)** | ~2 GB | ~4 GB | ~7-8 GB |
| **Training Speed** | Fast | Medium | Slow |
| **Reasoning Focus** | No | Moderate | **Strong** |
| **Chat Template** | Optional | **Required** | **Required** |
| **Special Features** | Baseline | Long context | Reasoning chains |
| **Best For** | Quick iteration | Balanced performance | Best accuracy |

## Model Details

### 1. Qwen/Qwen2.5-1.5B (Default/Baseline)

**Architecture**:
- Standard Qwen2.5 transformer
- Separate Q, K, V attention projections
- SwiGLU activation (gated FFN)
- RoPE positional encoding

**Strengths**:
- ✅ Smallest model (fastest training)
- ✅ Low memory requirements
- ✅ Good baseline performance
- ✅ Easy to debug and iterate
- ✅ No special chat template needed

**Weaknesses**:
- ❌ Lower accuracy than larger models
- ❌ Not specialized for reasoning
- ❌ Basic prompt formatting

**Files**:
- Config: `configs/sft_config.yaml`, `configs/grpo_config.yaml`
- Preprocessing: `scripts/data_preprocessing.py`
- SLURM: `submit_sft.slurm`, `submit_grpo.slurm`

**Usage**:
```bash
sbatch submit_sft.slurm
sbatch submit_grpo.slurm
```

---

### 2. microsoft/Phi-3.5-mini-instruct (Medium Model)

**Architecture**:
- Phi-3 transformer with combined projections
- **Combined QKV projection** (single `qkv_proj` layer)
- **Combined gate/up projection** (`gate_up_proj`)
- Long context support (128K tokens)

**Strengths**:
- ✅ Best size/performance balance
- ✅ Strong reasoning for 3.8B size
- ✅ Long context window (128K)
- ✅ Microsoft-backed model
- ✅ +10% accuracy over Qwen

**Weaknesses**:
- ❌ **Requires special chat template** (`<|user|>...<|end|><|assistant|>`)
- ❌ Different LoRA targets than Qwen
- ❌ More memory than Qwen
- ❌ Moderate training time

**Files**:
- Config: `configs/sft_config_phi3.yaml`, `configs/grpo_config_phi3.yaml`
- Preprocessing: `scripts/data_preprocessing_phi3.py`
- SLURM: `submit_sft_phi3.slurm`, `submit_grpo_phi3.slurm`
- Pipeline: `run_phi3_pipeline.sh`

**Usage**:
```bash
bash run_phi3_pipeline.sh
```

---

### 3. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (NEW - Largest Model)

**Architecture**:
- Based on Qwen2.5-Math-7B
- Fine-tuned with DeepSeek-R1 reasoning data
- **Pre-trained for chain-of-thought reasoning**
- Uses `<think>` tags for reasoning chains

**Strengths**:
- ✅ **Best reasoning performance** (pre-trained for reasoning)
- ✅ Highest expected accuracy (+15-20% over Qwen)
- ✅ Ideal for GRPO (reasoning-focused)
- ✅ Strong chain-of-thought capabilities
- ✅ Based on familiar Qwen architecture

**Weaknesses**:
- ❌ **Largest memory requirement** (7-8 GB)
- ❌ **Slowest training** (16-21 hours total)
- ❌ Requires chat template
- ❌ Requires special prompt formatting
- ❌ Longer generation times (reasoning chains)

**Files**:
- Config: `configs/sft_config_deepseek.yaml`, `configs/grpo_config_deepseek.yaml`
- Preprocessing: `scripts/data_preprocessing_deepseek.py`
- SLURM: `submit_sft_deepseek.slurm`, `submit_grpo_deepseek.slurm`
- Pipeline: `run_deepseek_pipeline.sh`
- Documentation: `DEEPSEEK_README.md`

**Usage**:
```bash
bash run_deepseek_pipeline.sh
```

---

## Architecture Comparison

### LoRA Target Modules

Different models require different LoRA targets:

**Qwen2.5 (1.5B) and DeepSeek (7B)**:
```yaml
lora_target_modules: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
```
- Separate Q, K, V projections in attention
- Separate gate and up projections in FFN

**Phi-3.5-mini (3.8B)**:
```yaml
lora_target_modules: "qkv_proj,o_proj,gate_up_proj,down_proj"
```
- **Combined QKV projection** (single layer)
- **Combined gate_up projection**
- More efficient but different layer names

### Chat Template Requirements

**Qwen2.5-1.5B**: No special template (optional)
```
Premise and Hypothesis:
{input}

Is the hypothesis consistent with the premise?
```

**Phi-3.5-mini**: Required special format
```
<|user|>
{prompt}
<|end|>
<|assistant|>
{response}
<|end|>
```

**DeepSeek-R1-7B**: Required with reasoning encouragement
```
<|user|>
{prompt}
Think step by step using <think> tags...
<|assistant|>
<think>
{reasoning}
</think>
{answer}
```

---

## Performance Comparison

### Expected Accuracy (Estimated)

| Model | SFT Accuracy | GRPO Accuracy | Improvement |
|-------|-------------|---------------|-------------|
| Qwen2.5-1.5B | ~72% | ~78% | +6% |
| Phi-3.5-mini | ~75% | ~85% | +10% |
| **DeepSeek-R1-7B** | **~78-80%** | **~88-92%** | **+10-12%** |

*Note: These are estimates based on model size and capabilities.*

### Training Time (2x RTX 2080 Ti, 11GB each)

| Stage | Qwen2.5-1.5B | Phi-3.5-mini | DeepSeek-R1-7B |
|-------|-------------|--------------|----------------|
| Preprocessing | 30 min | 45 min | 60 min |
| SFT Training | 3-4 hours | 5-6 hours | 6-8 hours |
| GRPO Training | 6-8 hours | 8-10 hours | 10-12 hours |
| **Total** | **9-12 hours** | **13-16 hours** | **16-21 hours** |

### Memory Usage (4-bit quantization)

| Model | Per-GPU Memory | Recommended Setup |
|-------|---------------|-------------------|
| Qwen2.5-1.5B | ~2 GB | 2x 6GB GPUs (easy) |
| Phi-3.5-mini | ~4 GB | 2x 8GB GPUs (moderate) |
| **DeepSeek-R1-7B** | **~7-8 GB** | **2x 11GB GPUs (tight)** |

---

## Configuration Comparison

### Batch Size and Learning Rate

| Model | Batch Size | Grad Accum | Effective Batch | Learning Rate (SFT) |
|-------|-----------|------------|----------------|-------------------|
| Qwen2.5-1.5B | 1 | 16 | 32 | 3.0e-5 |
| Phi-3.5-mini | 1 | 16 | 32 | 2.0e-5 |
| DeepSeek-R1-7B | 1 | 32 | 64 | 2.0e-5 |

### LoRA Configuration

| Model | LoRA Rank | LoRA Alpha | Rationale |
|-------|-----------|------------|-----------|
| Qwen2.5-1.5B | 32 | 16 | Smaller model needs less capacity |
| Phi-3.5-mini | 64 | 32 | Medium model benefits from more capacity |
| DeepSeek-R1-7B | 64 | 32 | Larger model needs higher rank |

---

## When to Use Each Model

### Use Qwen2.5-1.5B when:
- ✅ Rapid prototyping and debugging
- ✅ Limited GPU memory (<8 GB)
- ✅ Need quick baseline results
- ✅ Training time is critical
- ✅ Learning the codebase

### Use Phi-3.5-mini when:
- ✅ Want balanced size/performance
- ✅ Have 8-16 GB GPU memory
- ✅ Need better accuracy than Qwen
- ✅ Don't need cutting-edge reasoning
- ✅ Want Microsoft's optimizations

### Use DeepSeek-R1-7B when:
- ✅ **Need best possible accuracy**
- ✅ **Task requires strong reasoning**
- ✅ Have 11+ GB GPU memory
- ✅ Can afford longer training
- ✅ **Doing GRPO training** (reasoning-focused)
- ✅ Need chain-of-thought capabilities

---

## Running All Models

### Train All Models in Parallel

```bash
# Terminal 1: Qwen2.5-1.5B
sbatch submit_sft.slurm
# Wait for completion
sbatch submit_grpo.slurm

# Terminal 2: Phi-3.5-mini
bash run_phi3_pipeline.sh

# Terminal 3: DeepSeek-R1-7B
bash run_deepseek_pipeline.sh
```

### Evaluate All Models

```bash
bash evaluate_all_models.sh
```

This will evaluate all 6 model variants:
1. Qwen2.5-1.5B-SFT
2. Qwen2.5-1.5B-GRPO
3. Phi-3.5-mini-SFT
4. Phi-3.5-mini-GRPO
5. DeepSeek-R1-7B-SFT
6. DeepSeek-R1-7B-GRPO

---

## File Organization

```
RLcausality/
├── configs/
│   ├── sft_config.yaml              # Qwen2.5-1.5B SFT
│   ├── grpo_config.yaml             # Qwen2.5-1.5B GRPO
│   ├── sft_config_phi3.yaml         # Phi-3.5-mini SFT
│   ├── grpo_config_phi3.yaml        # Phi-3.5-mini GRPO
│   ├── sft_config_deepseek.yaml     # DeepSeek-R1-7B SFT (NEW)
│   └── grpo_config_deepseek.yaml    # DeepSeek-R1-7B GRPO (NEW)
│
├── scripts/
│   ├── data_preprocessing.py        # Qwen (generic)
│   ├── data_preprocessing_phi3.py   # Phi-3 with chat template
│   └── data_preprocessing_deepseek.py  # DeepSeek with reasoning (NEW)
│
├── submit_*.slurm                   # SLURM job files
│   ├── submit_sft.slurm             # Qwen SFT
│   ├── submit_grpo.slurm            # Qwen GRPO
│   ├── submit_sft_phi3.slurm        # Phi-3 SFT
│   ├── submit_grpo_phi3.slurm       # Phi-3 GRPO
│   ├── submit_sft_deepseek.slurm    # DeepSeek SFT (NEW)
│   └── submit_grpo_deepseek.slurm   # DeepSeek GRPO (NEW)
│
├── run_*.sh                         # Convenience scripts
│   ├── run_phi3_pipeline.sh         # Full Phi-3 pipeline
│   ├── run_deepseek_pipeline.sh     # Full DeepSeek pipeline (NEW)
│   ├── evaluate_both_models.sh      # Eval Qwen models
│   └── evaluate_all_models.sh       # Eval all 6 models (NEW)
│
└── *.md                             # Documentation
    ├── DEEPSEEK_README.md           # DeepSeek guide (NEW)
    └── MODEL_COMPARISON.md          # This file (NEW)
```

---

## Recommendations

### For Research and Publication
**Use all three models** to show performance scaling:
- Qwen2.5-1.5B: Baseline/small model performance
- Phi-3.5-mini: Medium model performance
- DeepSeek-R1-7B: State-of-the-art reasoning performance

### For Production
**Use DeepSeek-R1-7B** if:
- Accuracy is critical
- You have sufficient compute
- Task involves complex reasoning

**Use Phi-3.5-mini** if:
- Need good balance of speed and accuracy
- Want Microsoft's optimization

**Use Qwen2.5-1.5B** if:
- Inference speed is critical
- Deploying on edge devices
- Memory is constrained

### For Development
**Start with Qwen2.5-1.5B** for fast iteration, then **scale up to DeepSeek-R1-7B** for final results.

---

## Expected Results Summary

Based on model capabilities, we expect the following ranking:

**Accuracy (High to Low)**:
1. 🥇 DeepSeek-R1-7B + GRPO (~88-92%)
2. 🥈 Phi-3.5-mini + GRPO (~85%)
3. 🥉 DeepSeek-R1-7B + SFT (~78-80%)
4. Qwen2.5-1.5B + GRPO (~78%)
5. Phi-3.5-mini + SFT (~75%)
6. Qwen2.5-1.5B + SFT (~72%)

**Training Speed (Fast to Slow)**:
1. 🥇 Qwen2.5-1.5B (9-12 hours)
2. 🥈 Phi-3.5-mini (13-16 hours)
3. 🥉 DeepSeek-R1-7B (16-21 hours)

**Memory Efficiency (Low to High)**:
1. 🥇 Qwen2.5-1.5B (~2 GB)
2. 🥈 Phi-3.5-mini (~4 GB)
3. 🥉 DeepSeek-R1-7B (~7-8 GB)

---

## Support and Issues

- **Qwen issues**: Check standard configuration files
- **Phi-3 issues**: See `run_phi3_pipeline.sh` and related files
- **DeepSeek issues**: See [DEEPSEEK_README.md](DEEPSEEK_README.md)
- **General issues**: Check logs in `logs/` directory

## References

- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Phi-3.5 Model Card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
