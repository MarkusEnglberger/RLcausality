"""
GRPO (Group Relative Policy Optimization) training script for reasoning with multi-GPU support.
"""

import os
import re
import torch
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
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
import wandb


@dataclass
class ScriptArguments:
    """
    Custom arguments for GRPO training (not part of GRPOConfig).
    Python dataclasses require all fields to have defaults if any field has a default.
    """

    # Model arguments
    model_name_or_path: str = field(
        default="./models/sft_model",
        metadata={"help": "Path to SFT model"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation: 'eager', 'flash_attention_2', or 'sdpa'"}
    )

    # Data arguments
    dataset_path: str = field(
        default="./data/processed/grpo",
        metadata={"help": "Path to processed GRPO dataset"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples"}
    )

    # Quantization arguments
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )

    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA R dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA"}
    )


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


def create_model_and_tokenizer(args: ScriptArguments):
    """Create and configure model and tokenizer."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # Important for generation
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if SFT model is a PEFT model
    is_peft_model = os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json"))

    if is_peft_model:
        print("Detected PEFT adapter in SFT model. Loading base model + adapter...")

        # Load adapter config to get base model name
        import json
        with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path", args.model_name_or_path)

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
        }

        # Add quantization config
        if args.use_4bit:
            from transformers import BitsAndBytesConfig
            print("Using 4-bit quantization (NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.use_8bit:
            from transformers import BitsAndBytesConfig
            print("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["dtype"] = torch.bfloat16

        # Set attention implementation
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        elif args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load base model
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )

        # Load PEFT adapter
        print(f"Loading PEFT adapter from: {args.model_name_or_path}")
        model = PeftModel.from_pretrained(model, args.model_name_or_path)

        # Optionally merge adapter weights for faster inference
        # model = model.merge_and_unload()

    else:
        print("Loading model (not a PEFT model)...")
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
        }

        # Add quantization config
        if args.use_4bit:
            from transformers import BitsAndBytesConfig
            print("Using 4-bit quantization (NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.use_8bit:
            from transformers import BitsAndBytesConfig
            print("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["dtype"] = torch.bfloat16

        # Set attention implementation
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        elif args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model directly
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )

    # Prepare model for k-bit training if using quantization
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training (enables gradient checkpointing with quantization)")

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
    elif args.use_lora:
        # Model doesn't have PEFT, add new LoRA layers
        from peft import get_peft_model

        target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else None

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    # Parse arguments
    import sys
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))

    # Require YAML config file
    if len(sys.argv) < 2 or not sys.argv[1].endswith('.yaml'):
        print("Error: YAML config file is required!")
        print("Usage: python train_grpo.py <config.yaml> [--arg value ...]")
        print("Example: python train_grpo.py configs/grpo_config.yaml")
        sys.exit(1)

    # Load YAML and any CLI overrides
    script_args, training_args = parser.parse_yaml_file(
        yaml_file=sys.argv[1],
        allow_extra_keys=True
    )

    # If there are additional CLI arguments, parse them as overrides
    if len(sys.argv) > 2:
        # Re-parse with CLI args as overrides
        cli_dict = {}
        i = 2
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                key = sys.argv[i][2:].replace('-', '_')
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    cli_dict[key] = sys.argv[i + 1]
                    i += 2
                else:
                    cli_dict[key] = True
                    i += 1
            else:
                i += 1
        # Apply CLI overrides to training_args
        for key, value in cli_dict.items():
            if hasattr(training_args, key):
                setattr(training_args, key, value)
            elif hasattr(script_args, key):
                setattr(script_args, key, value)

    # Set seed
    set_seed(training_args.seed)

    # Initialize wandb
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project=getattr(training_args, 'wandb_project', 'corr2cause-grpo'),
            name=training_args.run_name,
            config={**vars(script_args), **vars(training_args)}
        )

    # Load dataset
    print(f"Loading dataset from {script_args.dataset_path}...")
    dataset = load_from_disk(script_args.dataset_path)

    # Rename 'query' to 'prompt' (GRPO expects 'prompt' field)
    def rename_query_to_prompt(example):
        example['prompt'] = example.pop('query')
        return example

    print("Renaming 'query' field to 'prompt' for GRPO compatibility...")
    dataset = dataset.map(rename_query_to_prompt)

    # Limit training samples if specified
    if script_args.max_train_samples is not None:
        original_train_size = len(dataset['train'])
        if original_train_size > script_args.max_train_samples:
            print(f"Limiting training samples from {original_train_size} to {script_args.max_train_samples}")
            dataset['train'] = dataset['train'].select(range(script_args.max_train_samples))

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    # Create model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(script_args)

    # Custom reward function that includes labels
    # GRPO calls reward functions with keyword arguments:
    # reward_func(prompts=prompts, completions=completions, completions_ids=completions_ids, **dataset_columns)
    # Dataset columns (like 'label') are passed as kwargs
    def reward_fn(completions, label, prompts=None, completions_ids=None, **kwargs):
        """Calculate rewards based on answer correctness."""
        # 'label' comes directly from the dataset as a keyword argument
        # 'completions' contains the generated text
        return calculate_rewards(completions, label)

    # Verify trainable parameters one more time before training
    print("\n=== Final Parameter Verification Before Training ===")
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            if trainable_count <= 5:  # Show first 5 trainable params
                print(f"  ✓ Trainable: {name}")
        else:
            frozen_count += 1
    print(f"Total trainable parameters: {trainable_count}")
    print(f"Total frozen parameters: {frozen_count}")
    print("=" * 55)

    # Initialize GRPO Trainer
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
