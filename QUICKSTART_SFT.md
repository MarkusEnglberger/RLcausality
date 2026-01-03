# Quick Start: SFT Training with LoRA and 4-bit Quantization

## TL;DR

Finetune DeepSeek 32B on 96 correct GPT reasoning traces using LoRA with 4-bit quantization.

```bash
# Submit SLURM job (recommended)
sbatch sft_train.slurm

# Or run locally
bash scripts/run_sft_pipeline.sh
```

## What This Does

1. **Filters correct traces**: Extracts 96 correct reasoning samples from GPT evaluation results
2. **Trains DeepSeek 32B**: Uses LoRA + 4-bit quantization for memory-efficient finetuning
3. **Saves adapter**: Stores trained LoRA adapter to `./models/deepseek_sft_lora_4bit`

## Key Features

- **Memory Efficient**: Uses only ~25GB VRAM (vs 80GB+ without quantization)
- **Fast Training**: ~1-2 hours on H100 GPU
- **Small Adapter**: Only ~1% of parameters are trainable
- **Easy Loading**: Simple PEFT integration for inference

## Files Overview

| File | Purpose |
|------|---------|
| [scripts/prepare_sft_data.py](scripts/prepare_sft_data.py) | Filters correct GPT traces |
| [scripts/train_sft_lora.py](scripts/train_sft_lora.py) | SFT training script |
| [configs/sft_config_deepseek.yaml](configs/sft_config_deepseek.yaml) | Hyperparameters |
| [sft_train.slurm](sft_train.slurm) | SLURM job script |

## Training Configuration Summary

```yaml
Model: DeepSeek-R1-Distill-Qwen-32B
Quantization: 4-bit NF4 with nested quant
LoRA Rank: 16
LoRA Alpha: 32
Learning Rate: 2e-5
Batch Size: 1 (effective 8 with grad accumulation)
Epochs: 3
Max Seq Length: 4096
Optimizer: paged_adamw_8bit
```

## Expected Results

- **Training Samples**: 82 correct GPT traces
- **Validation Samples**: 14 correct GPT traces
- **Training Steps**: ~30 per epoch × 3 epochs = ~90 steps
- **Time**: 1-2 hours on H100

## After Training

### Evaluate the Model

```bash
# Update evaluation config
# Edit configs/evaluation_config.yaml:
#   model_path: "./models/deepseek_sft_lora_4bit"

python scripts/evaluate_model_simple.py
```

### Use in Code

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base + adapter
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, "./models/deepseek_sft_lora_4bit")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)
```

## Monitor Training

Training logs to WandB automatically:
- Project: `corr2cause-sft`
- Run name: `deepseek-32b-sft-lora-4bit`

## Troubleshooting

**OOM Error**: Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps`

**Slow Training**: Normal for 32B model. Expected ~1-2 hours on H100.

**WandB Issues**: Set `report_to: "none"` in config to disable

## Full Documentation

See [SFT_TRAINING_README.md](SFT_TRAINING_README.md) for complete documentation.
