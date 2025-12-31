"""
Data preprocessing script for DeepSeek-R1-Distill-Qwen-7B.

DeepSeek-R1 is trained for reasoning and uses special formatting:
- Avoid system prompts (use user prompts only)
- Encourage reasoning with <think> tags
- Temperature: 0.5-0.7 (0.6 recommended)
- Based on Qwen2.5 architecture with chat template support
"""

import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer


def format_grpo_query_deepseek(tokenizer, input_text: str) -> str:
    """
    Format GRPO query using DeepSeek-R1's chat template with reasoning encouragement.
    DeepSeek-R1 is pre-trained for reasoning, so we encourage <think> tags.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert in causal reasoning. Given a premise describing "
                "statistical relationships between variables, determine if the stated "
                "causal hypothesis is consistent with the given information.\n\n"
                "After your reasoning, provide your final answer as either \"Therefore: Yes\" or \"Therefore: No\".\n\n"
                f"Premise and Hypothesis:\n{input_text}\n\n"
            )
        }
    ]

    # Use chat template with generation prompt
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted



def preprocess_for_grpo(args):
    """Preprocess dataset for GRPO training with DeepSeek-R1-Distill-Qwen-7B."""
    print("=" * 80)
    print("Preprocessing Corr2Cause dataset for GRPO with DeepSeek-R1-Distill-Qwen-7B")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from HuggingFace...")
    dataset = load_dataset(
        'causal-nlp/corr2cause',
        cache_dir=args.cache_dir
    )

    # Process function
    def process_grpo_example(example):
        """Process single example for GRPO (query only, no answer)."""
        input_text = example['input']

        # Format using DeepSeek-R1 chat template with reasoning encouragement
        query = format_grpo_query_deepseek(tokenizer, input_text)

        return {
            'query': query,
            'label': example['label'],
            'num_variables': example['num_variables'],
            'template': example['template']
        }

    # Process all splits
    print("\nProcessing GRPO dataset...")
    processed_dataset = dataset.map(
        process_grpo_example,
        remove_columns=['input'],
        desc="Formatting for GRPO"
    )

    # Save processed dataset
    output_path = os.path.join(args.output_dir, "grpo_deepseek")
    os.makedirs(output_path, exist_ok=True)

    print(f"\nSaving processed GRPO dataset to {output_path}...")
    processed_dataset.save_to_disk(output_path)
    print("GRPO dataset saved!")

    # Print example
    print("\n" + "=" * 80)
    print("Example GRPO formatted query:")
    print("=" * 80)
    print(processed_dataset['train'][0]['query'][:800] + "...\n")

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Corr2Cause dataset for DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name or path for tokenizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/cache",
        help="Cache directory for downloads"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["sft", "grpo", "both"],
        default="both",
        help="Which preprocessing stage to run"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)


    if args.stage in ["grpo", "both"]:
        grpo_dataset = preprocess_for_grpo(args)
        print(f"\nGRPO dataset sizes:")
        print(f"  Train: {len(grpo_dataset['train'])}")
        print(f"  Validation: {len(grpo_dataset['validation'])}")
        print(f"  Test: {len(grpo_dataset['test'])}")

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
