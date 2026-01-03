"""
Supervised Fine-Tuning (SFT) script for DeepSeek 32B with LoRA and 4-bit quantization.
Uses correct reasoning traces from GPT to finetune the model.
"""
import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig(TrainingArguments):
    """Extended TrainingArguments with custom SFT parameters."""

    # Model configuration
    model_name_or_path: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    # Data configuration
    dataset_path: str = field(default="./data/processed/sft_deepseek")
    max_train_samples: Optional[int] = field(default=None)

    # LoRA configuration
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")

    # Quantization configuration
    use_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    use_nested_quant: bool = field(default=True)

    # Sequence length
    max_seq_length: int = field(default=4096)

    # WandB configuration
    wandb_project: str = field(default="corr2cause-sft") 


def create_peft_config(training_args: SFTConfig):
    """Create PEFT (LoRA) configuration."""
    target_modules = training_args.lora_target_modules.split(",")

    peft_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


def load_model_and_tokenizer(training_args: SFTConfig):
    """Load model with 4-bit quantization and tokenizer."""
    print(f"Loading model: {training_args.model_name_or_path}")

    # Configure quantization if enabled
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if training_args.use_4bit:
        bnb_config = BitsAndBytesConfig( load_in_4bit=training_args.use_4bit,
                                        bnb_4bit_quant_type=training_args.bnb_4bit_quant_type,
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_use_double_quant=training_args.use_nested_quant,)
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        **model_kwargs
    )

    # Prepare model for k-bit training if using quantization
    if training_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",  # Use right padding for training
    )

    return model, tokenizer


def formatting_prompts_func(examples):
    output_texts = []
    for query, response in zip(examples["query"], examples["response"]):
        # Format as a conversation with clear delimiters
        text = f"{query}\n\n{response}"
        output_texts.append(text)
    return output_texts


def main():
    # Parse arguments
    parser = HfArgumentParser(SFTConfig)
    training_args = parser.parse_yaml_file(yaml_file=sys.argv[1])[0]

    # Initialize WandB if enabled
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name,
            config=vars(training_args)
        )

    # Load dataset
    print(f"\nLoading dataset from {training_args.dataset_path}...")
    dataset = load_from_disk(training_args.dataset_path)

    # Limit training samples if specified
    if training_args.max_train_samples is not None:
        dataset['train'] = dataset['train'].select(range(min(training_args.max_train_samples, len(dataset['train']))))
        print(f"Limited training to {len(dataset['train'])} samples")

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(training_args)

    # Apply LoRA if enabled
    if training_args.use_lora:
        print("\nApplying LoRA configuration...")
        peft_config = create_peft_config(training_args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Create trainer
    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=training_args.max_seq_length,
        packing=False,  # Don't pack sequences for reasoning tasks
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
