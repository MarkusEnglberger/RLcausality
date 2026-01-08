import os
import sys
from transformers import (HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
import torch
from peft import get_peft_model
from typing import Optional, List
import wandb
import json

def extract_answer(text: str) -> Optional[bool]:
    """Extract the answer from generated text."""
    text_lower = text.lower()
    if "therefore: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower:
        return False
    return None

def reward_fn(prompts: List[str], completions: List[str], label: List[int], **kwargs) -> List[float]:
    rewards = []
    for completion, true_label in zip(completions, label):
        pred = extract_answer(completion)
        if pred is None:
            reward = -0.5 - len(completion) * 0.00001
        elif pred == bool(true_label):
            reward = 1.0 - len(completion) * 0.00001
        else:
            reward = -1.0 - len(completion) * 0.00001
        rewards.append(reward)
    return rewards

def create_model_and_tokenizer(training_args, extra_args):
    model_name_or_path = extra_args.get('model_name_or_path')
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")

    # Configure model loading with optional 4-bit quantization
    model_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}

    if extra_args.get('use_4bit', False):
        print("Using 4-bit quantization (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    is_peft_model = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft_model:
        print("Model has existing PEFT adapters - merging into base model for GRPO")
        with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        model_name = adapter_config.get("base_model_name_or_path", model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name,**model_kwargs)
        model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=False)

        # Merge adapters into base model to ensure reference model is properly initialized
        print("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        print("Adapters merged successfully")

        # Now add new LoRA adapters for GRPO training
        if extra_args.get('use_lora', False):
            if extra_args.get('use_4bit', False):
                model = prepare_model_for_kbit_training(model)
            target_modules = extra_args.get('lora_target_modules', 'q_proj,v_proj').split(",")
            peft_config = LoraConfig(
                r=extra_args.get('lora_r', 16),
                lora_alpha=extra_args.get('lora_alpha', 16),
                lora_dropout=extra_args.get('lora_dropout', 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    elif extra_args.get('use_lora', False):
        model=AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        # Prepare model for k-bit training if using quantization
        if extra_args.get('use_4bit', False):
            model = prepare_model_for_kbit_training(model)
        target_modules = extra_args.get('lora_target_modules', 'q_proj,v_proj').split(",")
        peft_config = LoraConfig(r=extra_args.get('lora_r', 16),
            lora_alpha=extra_args.get('lora_alpha', 16),
            lora_dropout=extra_args.get('lora_dropout', 0.05),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model=AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    return model, tokenizer
    
def main():
    parser = HfArgumentParser(GRPOConfig)
    training_args = parser.parse_yaml_file(yaml_file=sys.argv[1], allow_extra_keys=True)[0]

    # Hardcoded paths and settings
    #model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    model_name_or_path = "./models/deepseek_sft_lora_4bit"
    dataset_path = "./data/processed/grpo"
    max_train_samples = 10000

    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project='corr2cause-grpo',
            name=getattr(training_args, 'run_name', None),
            config=vars(training_args)
        )

    dataset=load_from_disk(dataset_path)

    # Rename 'query' to 'prompt' for GRPO trainer compatibility
    dataset = dataset.rename_column('query', 'prompt')

    if max_train_samples is not None:
        dataset['train'] = dataset['train'].select(range(max_train_samples))

    # Hardcoded LoRA and quantization settings for extra_args
    extra_args = {
        'model_name_or_path': model_name_or_path,
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_target_modules': 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
        'use_4bit': True
    }
    model, tokenizer = create_model_and_tokenizer(training_args, extra_args)

    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],  # Must be a list of reward functions
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"].select(range(100, 111)),  # Only evaluate on samples 100-110
    )

    # Ensure value head is in the correct dtype for 4-bit training
    if extra_args.get('use_4bit', False) and hasattr(trainer.model, 'v_head'):
        trainer.model.v_head = trainer.model.v_head.to(torch.bfloat16)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    

if __name__ == "__main__":
    main()

