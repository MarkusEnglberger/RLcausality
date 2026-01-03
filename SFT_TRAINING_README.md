# Supervised Fine-Tuning (SFT) with LoRA and 4-bit Quantization

This guide explains how to finetune the DeepSeek 32B model using LoRA with 4-bit quantization on correct reasoning traces from GPT.

## Overview

The training pipeline consists of two main steps:

1. **Data Preparation**: Filter correct reasoning traces from GPT evaluation results
2. **SFT Training**: Finetune DeepSeek 32B with LoRA and 4-bit quantization

## Files Created

### Scripts
- [scripts/prepare_sft_data.py](scripts/prepare_sft_data.py) - Filters correct GPT traces and prepares dataset
- [scripts/train_sft_lora.py](scripts/train_sft_lora.py) - SFT training script with LoRA and 4-bit quantization

### Configuration
- [configs/sft_config_deepseek.yaml](configs/sft_config_deepseek.yaml) - Training hyperparameters

### SLURM Job
- [sft_train.slurm](sft_train.slurm) - Complete pipeline job script

## Training Configuration

### Model Configuration
- **Base Model**: DeepSeek-R1-Distill-Qwen-32B
- **Quantization**: 4-bit (NF4) with nested quantization
- **Memory**: ~16GB VRAM (vs ~64GB unquantized)

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32 (2x rank)
- **Dropout**: 0.05
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable Parameters**: ~1% of total model parameters

### Training Hyperparameters
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps (effective batch size = 8)
- **Learning Rate**: 2e-5
- **Epochs**: 3 (with 96 correct samples = ~300 training steps)
- **Optimizer**: paged_adamw_8bit (memory efficient)
- **Max Sequence Length**: 4096 tokens
- **Gradient Checkpointing**: Enabled

## Usage

### Quick Start

Submit the SLURM job to run the complete pipeline:

```bash
sbatch sft_train.slurm
```

This will:
1. Prepare the dataset from GPT predictions
2. Train the model with LoRA and 4-bit quantization
3. Save the trained adapter to `./models/deepseek_sft_lora_4bit`

### Manual Step-by-Step

If you want to run steps manually:

#### Step 1: Prepare Dataset

```bash
python scripts/prepare_sft_data.py
```

This creates:
- `./data/processed/sft_deepseek/` - HuggingFace dataset
- `./data/processed/sft_deepseek/sft_data_summary.json` - Summary statistics

#### Step 2: Train Model

```bash
python scripts/train_sft_lora.py configs/sft_config_deepseek.yaml
```

Or with custom arguments:

```bash
python scripts/train_sft_lora.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --dataset_path ./data/processed/sft_deepseek \
    --output_dir ./models/deepseek_sft_lora_4bit \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_lora true \
    --use_4bit true \
    --lora_r 16 \
    --lora_alpha 32
```

## Loading the Finetuned Model

After training, load the model with the LoRA adapter:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Setup 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    "./models/deepseek_sft_lora_4bit"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    trust_remote_code=True
)

# Use the model
prompt = "Your causal reasoning query here..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Dataset Information

The training dataset is created from GPT evaluation results:

- **Source**: `evaluation_results/gpt_predictions_gpt-5.2_20260101_195905.json`
- **Total GPT samples**: 100
- **Correct samples**: 96 (96% accuracy)
- **Train/Val Split**: 85/15 split
- **Train samples**: ~82
- **Validation samples**: ~14

Each training example consists of:
- **Query**: Original Corr2Cause causal reasoning problem
- **Response**: GPT's correct reasoning trace with step-by-step explanation
- **Label**: Ground truth causal answer

## Memory Requirements

With 4-bit quantization and LoRA:
- **Model Loading**: ~16GB VRAM
- **Training Peak**: ~25-30GB VRAM (with gradient checkpointing)
- **Without Quantization**: Would require ~80GB+ VRAM

## Monitoring Training

The training script logs to WandB by default. View metrics at:
- **Project**: corr2cause-sft
- **Run Name**: deepseek-32b-sft-lora-4bit

Logged metrics include:
- Training/validation loss
- Learning rate schedule
- GPU memory usage
- Samples per second

## Expected Training Time

On H100 GPU:
- **Dataset preparation**: ~1 minute
- **Model loading**: ~5-10 minutes
- **Training (3 epochs)**: ~1-2 hours
- **Total**: ~2-2.5 hours

## Evaluation

To evaluate the finetuned model:

```bash
# Update evaluation config to use the finetuned model
# Edit configs/evaluation_config.yaml:
#   model_path: "./models/deepseek_sft_lora_4bit"

python scripts/evaluate_model_simple.py
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce batch size in [configs/sft_config_deepseek.yaml](configs/sft_config_deepseek.yaml):
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16  # Increase to maintain effective batch size
   ```

2. Reduce sequence length:
   ```yaml
   max_seq_length: 2048  # Down from 4096
   ```

3. Use more aggressive quantization:
   ```yaml
   use_nested_quant: true
   bnb_4bit_compute_dtype: "float16"  # Instead of bfloat16
   ```

### CUDA Out of Memory During Loading

If the model fails to load:

```bash
# Clear CUDA cache before running
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### WandB Login Issues

If WandB authentication fails:

```bash
# Login to WandB before submitting job
wandb login

# Or disable WandB in config
report_to: "none"
```

## Next Steps

After SFT training, you can:

1. **Evaluate Performance**: Compare with base model using evaluate_model_simple.py
2. **Further Training**: Use GRPO on the SFT model for reinforcement learning
3. **Inference**: Deploy the finetuned model for causal reasoning tasks
4. **Merge Adapter**: Optionally merge LoRA adapter into base model for faster inference

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [TRL Documentation](https://huggingface.co/docs/trl)
