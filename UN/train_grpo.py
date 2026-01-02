"""
GRPO (Group Relative Policy Optimization) training script for reasoning with multi-GPU support.
"""

import os
import re
import json
import torch
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig
import wandb



def extract_answer(text: str) -> Optional[bool]:
    """Extract the answer from generated text."""
    text_lower = text.lower()

    # Look for explicit answer patterns
    if "therefore: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower:
        return False
    return None


def calculate_rewards(samples: List[str], labels: List[int]) -> List[float]:
    """
    Calculate rewards for generated samples based on correctness.

    Args:
        samples: List of generated text responses
        labels: List of ground truth labels (0 or 1)

    Returns:
        List of reward values
    """
    return [
        -0.5 if (pred := extract_answer(sample)) is None
        else (1.0 if pred == bool(label) else -1.0)
        for sample, label in zip(samples, labels)
    ]


def create_model_and_tokenizer(training_args):
    """Create and configure model and tokenizer."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # Important for generation
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if SFT model is a PEFT model
    is_peft_model = os.path.exists(os.path.join(training_args.model_name_or_path, "adapter_config.json"))

    if is_peft_model:
        print("Loading base model + adapter...")

        # Load adapter config to get base model name
        with open(os.path.join(training_args.model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path", training_args.model_name_or_path)

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        # Set attention implementation
        if hasattr(training_args, 'use_flash_attention') and training_args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load base model
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )

        # Load PEFT adapter
        print(f"Loading PEFT adapter from: {training_args.model_name_or_path}")
        model = PeftModel.from_pretrained(model, training_args.model_name_or_path, is_trainable=True)

    else:
        print("Loading model (not a PEFT model)...")
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        # Set attention implementation
        if hasattr(training_args, 'use_flash_attention') and training_args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model directly
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            **model_kwargs
        )

    # Ensure PEFT adapters are trainable for GRPO
    if hasattr(model, "peft_config"):
        print("Model has existing PEFT adapters from SFT. Ensuring they are trainable...")

        # Enable gradients on all LoRA parameters
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

        # Verify at least some parameters are trainable
        if trainable_params == 0:
            raise RuntimeError("ERROR: No trainable parameters found! All gradients are disabled.")
    elif hasattr(training_args, 'use_lora') and training_args.use_lora:
        # Model doesn't have PEFT, add new LoRA layers
        from peft import get_peft_model

        target_modules = training_args.lora_target_modules.split(",") if hasattr(training_args, 'lora_target_modules') and training_args.lora_target_modules else None

        peft_config = LoraConfig(
            r=getattr(training_args, 'lora_r', 16),
            lora_alpha=getattr(training_args, 'lora_alpha', 16),
            lora_dropout=getattr(training_args, 'lora_dropout', 0.05),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    # Parse arguments
    parser = HfArgumentParser(GRPOConfig)

    # Require YAML config file
    if len(sys.argv) < 2 or not sys.argv[1].endswith('.yaml'):
        print("Error: YAML config file is required!")
        print("Usage: python train_grpo.py <config.yaml>")
        print("Example: python train_grpo.py configs/grpo_config.yaml")
        sys.exit(1)

    # Load YAML config
    training_args = parser.parse_yaml_file(yaml_file=sys.argv[1], allow_extra_keys=True)[0]

    # Set seed
    set_seed(training_args.seed)

    # Initialize wandb
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project=getattr(training_args, 'wandb_project', 'corr2cause-grpo'),
            name=getattr(training_args, 'run_name', None),
            config=vars(training_args)
        )

    # Load dataset
    dataset_path = getattr(training_args, 'dataset_path')
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Rename 'query' to 'prompt' (GRPO expects 'prompt' field)
    def rename_query_to_prompt(example):
        example['prompt'] = example.pop('query')
        return example

    print("Renaming 'query' field to 'prompt' for GRPO compatibility...")
    dataset = dataset.map(rename_query_to_prompt)

    # Limit training samples if specified
    max_train_samples = getattr(training_args, 'max_train_samples', None)
    if max_train_samples is not None:
        original_train_size = len(dataset['train'])
        if original_train_size > max_train_samples:
            print(f"Limiting training samples from {original_train_size} to {max_train_samples}")
            dataset['train'] = dataset['train'].select(range(max_train_samples))

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    # Create model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(training_args)

    # Custom reward function that includes labels
    # GRPO calls reward functions with keyword arguments:
    # reward_func(prompts=prompts, completions=completions, completions_ids=completions_ids, **dataset_columns)
    # Dataset columns (like 'label') are passed as kwargs
    def reward_fn(completions, label, **_kwargs):
        """Calculate rewards based on answer correctness."""
        # 'label' comes directly from the dataset as a keyword argument
        # 'completions' contains the generated text
        return calculate_rewards(completions, label)

    # Initialize GRPO Trainer
    # Note: Sample logging is now handled by TRL's built-in log_completions feature
    # Configure via 'log_completions: true' in the YAML config
    # Note: GRPOTrainer requires 'reward_funcs' (plural) as a required argument
    # The trainer will create a frozen reference model internally - warnings about
    # gradients from the reference model are expected and can be ignored
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],  # Must be a list of reward functions
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Train
    print("\nStarting GRPO training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("\nGRPO training complete!")


if __name__ == "__main__":
    main()
