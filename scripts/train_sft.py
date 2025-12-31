"""
Supervised Fine-Tuning (SFT) script with multi-GPU support using TRL.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ScriptArguments:
    """Arguments for SFT training."""

    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"}
    )

    # Data arguments
    dataset_path: str = field(
        default="./data/processed/sft",
        metadata={"help": "Path to processed dataset"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use (None = use all)"}
    )

    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA R dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )

    # Quantization
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization"}
    )


def create_model_and_tokenizer(args: ScriptArguments):
    """Create and configure model and tokenizer."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    # Add pad token if missing (Qwen models typically have pad token, but double check)
    if tokenizer.pad_token is None:
        # For Qwen, use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine compute dtype from training args (will be set later, but we need to know now for quantization)
    # For quantization, we want to match the training precision
    compute_dtype = torch.float16  # Default to fp16 for compatibility

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Quantization config
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.use_8bit:
        model_kwargs["load_in_8bit"] = True

    # Load model
    print(f"Loading model with config: {model_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    # Debug: Print model info to verify quantization
    print(f"\n=== Model Info ===")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    if hasattr(model.config, 'quantization_config'):
        print(f"Quantization config: {model.config.quantization_config}")
    else:
        print("WARNING: No quantization config found!")

    # Check first layer dtype to verify quantization
    first_layer = next(model.parameters())
    print(f"First layer dtype: {first_layer.dtype}")
    print(f"First layer device: {first_layer.device}")
    print(f"=================\n")

    # Prepare for k-bit training if using quantization
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training")

    

    # Apply LoRA
    if args.use_lora:
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
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    # Check if a YAML config file is provided as the first argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')):
        # Parse YAML config file + any additional command-line overrides
        script_args, training_args = parser.parse_yaml_file(sys.argv[1], allow_extra_keys=True)
    else:
        # Parse command-line arguments only
        script_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Note: WandB is automatically initialized by TrainingArguments if report_to="wandb"
    # Set WANDB_PROJECT environment variable before running to specify project name

    # Load dataset
    print(f"Loading dataset from {script_args.dataset_path}...")
    dataset = load_from_disk(script_args.dataset_path)

    # Limit training samples if specified
    if script_args.max_train_samples is not None:
        print(f"Limiting training set to {script_args.max_train_samples} samples...")
        dataset["train"] = dataset["train"].select(range(min(script_args.max_train_samples, len(dataset["train"]))))

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Create model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(script_args)

    # Initialize SFT Trainer
    # Note: TRL 0.24+ uses 'processing_class' instead of 'tokenizer'
    # max_seq_length is handled by the tokenizer's model_max_length
    tokenizer.model_max_length = script_args.max_seq_length

    # Configure SFTConfig to use the "text" field from our pre-formatted dataset
    # This is compatible with completion_only_loss (default in TRL 0.24+)
    from trl import SFTConfig
    sft_config = SFTConfig(
        dataset_text_field="text",  # Use our pre-formatted "text" field
        **training_args.to_dict()  # Include all training arguments
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,  # Use SFTConfig instead of TrainingArguments
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,  # TRL 0.24+ parameter name
    )

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
