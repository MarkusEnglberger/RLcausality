import os 
import sys
from transformers import (HfArgumentParser, AutoTokenizer, AutoModelForCausalLM)
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig
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

def reward_fn(samples: List[str], labels: List[int], **_kwargs) -> List[float]:
    return [
        -0.5 if (pred := extract_answer(sample)) is None
        else (1.0 if pred == bool(label) else -1.0)
        for sample, label in zip(samples, labels)
    ]

def create_model_and_tokenizer(training_args):
    tokenizer=AutoTokenizer.from_pretrained(training_args.model_name_or_path, trust_remote_code=True, padding_side="left")
    model_kwargs = {"trust_remote_code": True,  "torch_dtype": torch.bfloat16, "device_map":"auto"}
    is_peft_model = os.path.exists(os.path.join(training_args.model_name_or_path, "adapter_config.json"))
    if is_peft_model:
        print("Model has existing PEFT adapters")
        with open(os.path.join(training_args.model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        model_name = adapter_config.get("base_model_name_or_path", training_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name,**model_kwargs)
        model = PeftModel.from_pretrained(model, training_args.model_name_or_path, is_trainable=True)
    elif hasattr(training_args, 'use_lora') and training_args.use_lora:
        model=AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path, **model_kwargs)
        target_modules = training_args.lora_target_modules.split(",")
        peft_config = LoraConfig(r=getattr(training_args, 'lora_r', 16),
            lora_alpha=getattr(training_args, 'lora_alpha', 16),
            lora_dropout=getattr(training_args, 'lora_dropout', 0.05),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model=AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path, **model_kwargs)
    return model, tokenizer
    
def main():
    parser = HfArgumentParser(GRPOConfig)
    training_args = parser.parse_yaml_file(yaml_file=sys.argv[1], allow_extra_keys=True)[0]
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project=getattr(training_args, 'wandb_project', 'corr2cause-grpo'),
            name=getattr(training_args, 'run_name', None),
            config=vars(training_args)
        )
    dataset_path = getattr(training_args, 'dataset_path')
    dataset=load_from_disk(dataset_path)
    max_train_samples = getattr(training_args, 'max_train_samples', None)
    dataset['train'] = dataset['train'].select(range(max_train_samples))
    model, tokenizer = create_model_and_tokenizer(training_args)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],  # Must be a list of reward functions
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    

if __name__ == "__main__":
    main()

