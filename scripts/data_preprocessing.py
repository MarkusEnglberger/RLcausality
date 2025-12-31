"""
Data preprocessing script for the causal-nlp/corr2cause dataset.
"""

from datasets import load_dataset
from typing import Dict, Optional
import os


class Corr2CauseDataProcessor:
    """Processor for the corr2cause datasetd."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "./data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_dataset(self):
        """Load the corr2cause dataset from HuggingFace."""
        print("Loading causal-nlp/corr2cause dataset...")
        dataset = load_dataset("causal-nlp/corr2cause", cache_dir=self.cache_dir)
        print(f"Dataset loaded: {dataset}")
        return dataset

    def format_for_sft(self, example: Dict) -> Dict:
        """
        Creates a prompt-completion pair for.
        Trains on binary labels only
        """

        # Format the input
        prompt = f"Premise and Hypothesis:\n{example['input']}\n\n"
        prompt += "Is the hypothesis consistent with the premise?"

        # Format the response with just the answer
        label_text = "Yes" if example['label'] == 1 else "No"
        completion = f"Answer: {label_text}"

        return {
            "prompt": prompt,
            "completion": completion,
            "text": f"{prompt}\n\n{completion}"
        }

    def format_for_grpo(self, example: Dict) -> Dict:
        """
        Format examples for GRPO (reasoning-focused).
        Returns prompt for model to generate reasoning traces.
        """
        instruction = (
            """You are an expert in causal reasoning. Given a premise describing 
            statistical relationships between variables, determine if the stated 
            causal hypothesis is implied by the given information. 
            Provide detailed step-by-step reasoning.
            Conclude with your final answer as either "Therefore: Yes" or "Therefore: No".\n\n"""
        )

        prompt = f"{instruction}Premise and Hypothesis:\n{example['input']}\n\n"
        prompt += "Think step by step and provide your reasoning:"

        # For GRPO, we need the ground truth label for reward calculation
        return {
            "query": prompt,
            "label": example['label'],
            "num_variables": example['num_variables'],
            "template": example['template']
        }

    def prepare_sft_dataset(self, dataset):
        """Prepare dataset for SFT training."""
        print("Formatting dataset for SFT...")

        # Apply formatting to all splits
        formatted_dataset = dataset.map(
            self.format_for_sft,
            remove_columns=dataset['train'].column_names,
            desc="Formatting for SFT"
        )

        return formatted_dataset

    def prepare_grpo_dataset(self, dataset):
        """Prepare dataset for GRPO training."""
        print("Formatting dataset for GRPO...")

        # Apply formatting to all splits
        formatted_dataset = dataset.map(
            self.format_for_grpo,
            remove_columns=['input'],  # Keep label and metadata for reward
            desc="Formatting for GRPO"
        )

        return formatted_dataset

    def save_processed_dataset(self, dataset, output_dir: str, stage: str):
        """Save processed dataset to disk."""
        save_path = os.path.join(output_dir, stage)
        os.makedirs(save_path, exist_ok=True)

        print(f"Saving {stage} dataset to {save_path}...")
        dataset.save_to_disk(save_path)
        print(f"Dataset saved successfully!")


def main():
    """Main function to process and save datasets."""
    processor = Corr2CauseDataProcessor(cache_dir="./data/cache")

    # Load raw dataset
    raw_dataset = processor.load_dataset()

    # Prepare for SFT
    sft_dataset = processor.prepare_sft_dataset(raw_dataset)
    processor.save_processed_dataset(sft_dataset, "./data/processed", "sft")

    # Prepare for GRPO
    grpo_dataset = processor.prepare_grpo_dataset(raw_dataset)
    processor.save_processed_dataset(grpo_dataset, "./data/processed", "grpo")

    print("\nDataset statistics:")
    print(f"SFT - Train: {len(sft_dataset['train'])}, Val: {len(sft_dataset['validation'])}, Test: {len(sft_dataset['test'])}")
    print(f"GRPO - Train: {len(grpo_dataset['train'])}, Val: {len(grpo_dataset['validation'])}, Test: {len(grpo_dataset['test'])}")

    # Print example
    print("\nExample SFT formatted data:")
    print(sft_dataset['train'][0]['text'][:500] + "...")


if __name__ == "__main__":
    main()
