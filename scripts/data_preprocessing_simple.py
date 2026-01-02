from datasets import load_dataset
from typing import Dict
import os

def format_for_grpo(example: Dict) -> Dict:
        instruction = (
            """Given a premise describing statistical relationships between variables, determine if the stated 
            causal hypothesis is implied by the given information. 
            Conclude with your final answer as either "Therefore: Yes" or "Therefore: No".\n\n"""
        )

        prompt = f"{instruction}. Premise and Hypothesis:\n{example['input']}\n\n"

        # For GRPO, we need the ground truth label for reward calculation
        return {"query": prompt, "label": example['label'], "num_variables": example['num_variables'],
            "template": example['template']}

def format_for_GPT(example: Dict) -> Dict:
        instruction = (
            """In the following, you have to decide whether the Hypothesis is implied by the Premise. You should roughly follow the PC-algorithm and show your reasoning. 
            Only give the answer at the end, ending your answer with either "Therefore: Yes" or "Therefore: No".\n\n""")
        prompt = f"{instruction}. Premise and Hypothesis:\n{example['input']}\n\n"

        # For GRPO, we need the ground truth label for reward calculation
        return {"query": prompt, "label": example['label'], "num_variables": example['num_variables'],
            "template": example['template']}


def main():
    cache_dir = "./data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    raw_dataset=load_dataset("causal-nlp/corr2cause", cache_dir=cache_dir)
    formatted_dataset_grpo=raw_dataset.map(format_for_grpo, remove_columns=['input'], desc="Formatting for GRPO")
    formatted_dataset_GPT=raw_dataset.map(format_for_GPT, remove_columns=['input'], desc="Formatting for GPT")
    save_path_grpo = os.path.join("./data/processed", "grpo")
    save_path_GPT = os.path.join("./data/processed", "GPT")
    os.makedirs(save_path_grpo, exist_ok=True)
    os.makedirs(save_path_GPT, exist_ok=True)
    formatted_dataset_grpo.save_to_disk(save_path_grpo)
    formatted_dataset_GPT.save_to_disk(save_path_GPT)

    print(f"GRPO - Train: {len(formatted_dataset_grpo['train'])}, Val: {len(formatted_dataset_grpo['validation'])}, Test: {len(formatted_dataset_grpo['test'])}")
    print("\nExample formatted data (GRPO):")
    print(formatted_dataset_grpo['train'][0]['query'][:500] + "...")
    print("\nExample formatted data (GPT):")
    print(formatted_dataset_GPT['train'][0]['query'][:500] + "...")
    

if __name__ == "__main__":
    main()
