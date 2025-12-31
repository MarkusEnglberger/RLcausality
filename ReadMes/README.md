# LLM Fine-tuning for Causal Reasoning

This project implements a two-stage fine-tuning pipeline for training language models on the **causal-nlp/corr2cause** dataset:
1. **Supervised Fine-Tuning (SFT)** - Initial training on causal reasoning tasks
2. **Group Relative Policy Optimization (GRPO)** - Reinforcement learning to improve reasoning quality

The training pipeline is built using [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) and supports **multi-GPU training** out of the box.

## Dataset

**causal-nlp/corr2cause** is a causal reasoning benchmark with ~208k examples that tests whether models can infer causal relationships from statistical constraints.

- **Task**: Binary classification of causal hypotheses given statistical premises
- **Splits**: Train (206k), Validation (1.08k), Test (1.16k)
- **Features**: Input premise-hypothesis pairs, binary labels, number of variables, template types

## Project Structure

```
RLcausality/
├── configs/
│   ├── sft_config.yaml          # SFT training configuration
│   ├── grpo_config.yaml         # GRPO training configuration
│   └── deepspeed_config.json    # DeepSpeed ZeRO configuration
├── scripts/
│   ├── data_preprocessing.py    # Dataset loading and preprocessing
│   ├── train_sft.py            # SFT training script
│   └── train_grpo.py           # GRPO training script
├── data/
│   ├── cache/                  # HuggingFace cache
│   └── processed/              # Processed datasets
│       ├── sft/                # SFT-formatted data
│       └── grpo/               # GRPO-formatted data
├── models/
│   ├── sft_model/              # SFT checkpoint output
│   └── grpo_model/             # GRPO checkpoint output
├── requirements.txt
├── run_sft.sh                  # SFT training launcher
└── run_grpo.sh                 # GRPO training launcher
```

## Installation

### 1. Clone the repository and navigate to it

```bash
cd RLcausality
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Flash Attention 2 for faster training

```bash
pip install flash-attn --no-build-isolation
```

### 5. Configure WandB (for logging)

```bash
wandb login
```

## Usage

### Step 1: Data Preprocessing

First, download and preprocess the dataset:

```bash
python scripts/data_preprocessing.py
```

This will:
- Download the causal-nlp/corr2cause dataset from HuggingFace
- Format it for SFT training (prompt-completion pairs with reasoning)
- Format it for GRPO training (prompts with labels for reward calculation)
- Save processed datasets to `./data/processed/`

### Step 2: Supervised Fine-Tuning (SFT)

#### Configuration

Edit [configs/sft_config.yaml](configs/sft_config.yaml) to customize:
- `model_name_or_path`: Base model (e.g., "meta-llama/Llama-2-7b-hf")
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation
- `num_train_epochs`: Number of epochs
- LoRA parameters, learning rate, etc.

#### Single GPU Training

```bash
python scripts/train_sft.py configs/sft_config.yaml
```

#### Multi-GPU Training

**Option 1: Using torchrun (recommended)**

```bash
torchrun --nproc_per_node=4 scripts/train_sft.py configs/sft_config.yaml
```

**Option 2: Using Accelerate**

```bash
# First, configure accelerate
accelerate config

# Then launch training
accelerate launch --num_processes=4 scripts/train_sft.py configs/sft_config.yaml
```

**Option 3: Using DeepSpeed (for large models)**

```bash
deepspeed --num_gpus=4 scripts/train_sft.py \
    configs/sft_config.yaml \
    --deepspeed configs/deepspeed_config.json
```

**Or use the provided script:**

```bash
bash run_sft.sh
```

### Step 3: GRPO Training (Reinforcement Learning)

After SFT completes, train with GRPO to improve reasoning quality.

#### Configuration

Edit [configs/grpo_config.yaml](configs/grpo_config.yaml) to customize:
- `model_name_or_path`: Path to SFT checkpoint (default: "./models/sft_model")
- `correct_reward`: Reward for correct answers (default: 1.0)
- `incorrect_reward`: Penalty for incorrect answers (default: -1.0)
- `reasoning_length_bonus`: Bonus per reasoning step (default: 0.1)
- Other GRPO hyperparameters

#### Multi-GPU Training

**Using torchrun:**

```bash
torchrun --nproc_per_node=4 scripts/train_grpo.py configs/grpo_config.yaml
```

**Using Accelerate:**

```bash
accelerate launch --num_processes=4 scripts/train_grpo.py configs/grpo_config.yaml
```

**Or use the provided script:**

```bash
bash run_grpo.sh
```

## Multi-GPU Training Details

### Automatic Data Parallel (DDP)

The training scripts automatically use PyTorch's Distributed Data Parallel (DDP) when multiple GPUs are detected. No code changes needed!

### Effective Batch Size

The effective batch size is calculated as:
```
Effective Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

Example with 4 GPUs:
- SFT: `4 × 4 × 4 = 64`
- GRPO: `2 × 8 × 4 = 64`

### Memory Optimization Techniques

1. **LoRA (Low-Rank Adaptation)**: Reduces trainable parameters by ~90%
2. **Gradient Checkpointing**: Trades compute for memory
3. **BFloat16 Mixed Precision**: Reduces memory usage by 2x
4. **DeepSpeed ZeRO**: Distributes optimizer states and gradients across GPUs
5. **4-bit/8-bit Quantization**: Further reduces memory (set `use_4bit: true` or `use_8bit: true`)

### Recommended GPU Configurations

| Model Size | GPUs | Settings |
|------------|------|----------|
| 7B | 1-2 | LoRA + bf16 |
| 7B | 4+ | LoRA + bf16 + gradient checkpointing |
| 13B | 2-4 | LoRA + bf16 + DeepSpeed ZeRO-2 |
| 13B | 4+ | LoRA + bf16 + DeepSpeed ZeRO-3 |
| 70B | 8+ | 4-bit + LoRA + DeepSpeed ZeRO-3 |

## GRPO Reward Function

The GRPO training uses a custom reward function that:

1. **Extracts the answer** from generated text (Yes/No for consistency)
2. **Checks correctness** against ground truth labels
3. **Rewards correct answers** (+1.0 by default)
4. **Penalizes incorrect answers** (-1.0 by default)
5. **Bonus for reasoning quality**: Additional reward based on number of reasoning steps (encourages detailed explanations)

You can customize rewards in [configs/grpo_config.yaml](configs/grpo_config.yaml).

## Monitoring Training

### WandB (Weights & Biases)

Training metrics are logged to WandB by default. View:
- Training/validation loss
- Learning rate schedule
- Gradient norms
- GRPO rewards and KL divergence
- GPU memory usage

### TensorBoard

Alternatively, use TensorBoard:

```bash
# In config, set: report_to: "tensorboard"
tensorboard --logdir ./models/sft_model/runs
```

## Inference Example

After training, use your fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/grpo_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = """You are an expert in causal reasoning. Given a premise describing statistical relationships between variables, determine if the stated causal hypothesis is consistent with the given information.

Premise and Hypothesis:
In a closed system of 4 variables A, B, C, D: A and B are correlated. C and D are independent given A and B. Hypothesis: A causes B.

Think step by step and provide your reasoning:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Enable 4-bit quantization: `use_4bit: true`
5. Use DeepSpeed ZeRO-3

### Slow Training

1. Enable Flash Attention 2: `use_flash_attention: true`
2. Increase `dataloader_num_workers`
3. Use `bf16: true` on Ampere GPUs (A100, H100)
4. Reduce evaluation frequency (`eval_steps`)

### Multi-GPU Issues

1. Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
2. Check GPU availability: `nvidia-smi`
3. Use `accelerate config` to set up distributed training
4. Set `NCCL_P2P_DISABLE=1` if having communication issues

## Citation

If you use this code or the corr2cause dataset, please cite:

```bibtex
@article{corr2cause2024,
  title={Causal Reasoning with Statistical Constraints},
  author={causal-nlp},
  journal={HuggingFace Datasets},
  year={2024}
}
```

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
