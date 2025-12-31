"""
Data preprocessing script specifically for Phi-3.5-Mini.
Phi-3.5 uses a different chat template than Qwen, so we need special formatting.
"""

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def format_sft_prompt_phi3(tokenizer, input_text: str) -> tuple[str, str]:
    """
    Format SFT data using Phi-3.5's chat template.
    Returns (prompt, completion) as separate strings for SFTTrainer.

    Phi-3.5 uses this format:
    <|user|>
    {user_message}<|end|>
    <|assistant|>
    {assistant_message}<|end|>
    """
    # Create prompt with user message only
    user_message = [
        {
            "role": "user",
            "content": (
                f"Given a premise describing statistical relationships between variables, "
                f"determine if the stated causal hypothesis is consistent with the given information.\n\n"
                f"Premise and Hypothesis:\n{input_text}\n\n"
                f"Is the hypothesis consistent with the premise? Answer with 'Yes' or 'No'."
            )
        }
    ]

    # Use Phi-3.5's chat template with generation prompt for the prompt part
    prompt = tokenizer.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)

    return prompt


def format_grpo_query_phi3(tokenizer, input_text: str) -> str:
    """
    Format GRPO query using Phi-3.5's chat template (no answer, just prompt).
    """
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert in causal reasoning. Given a premise describing "
                "statistical relationships between variables, determine if the stated "
                "causal hypothesis is consistent with the given information. "
                "Provide detailed step-by-step reasoning.\n"
                "Conclude with your final answer as either \"Therefore: Yes\" or \"Therefore: No\".\n\n"
                f"Premise and Hypothesis:\n{input_text}\n\n"
                "Think step by step and provide your reasoning:"
            )
        }
    ]

    # Use Phi-3.5's chat template with generation prompt
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted


def preprocess_for_sft(args):
    """Preprocess dataset for SFT training with Phi-3.5."""
    print("=" * 80)
    print("Preprocessing Corr2Cause dataset for SFT with Phi-3.5-Mini")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # Phi-3.5 requires pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load dataset
    print(f"\nLoading dataset from HuggingFace...")
    dataset = load_dataset(
        'causal-nlp/corr2cause',
        cache_dir=args.cache_dir
    )

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Format answers
    def format_answer(label: int) -> str:
        """Convert binary label to text answer."""
        return "Yes" if label == 1 else "No"

    # Process function
    def process_sft_example(example):
        """Process single example for SFT."""
        input_text = example['input']
        label = example['label']
        answer = format_answer(label)

        # Format using Phi-3.5 chat template
        # Returns prompt with <|user|>...<|end|><|assistant|> ready for completion
        prompt = format_sft_prompt_phi3(tokenizer, input_text)

        # Completion is just the answer + EOS token
        # SFTTrainer will concatenate prompt + completion
        completion = answer + tokenizer.eos_token

        return {
            'prompt': prompt,
            'completion': completion,
            'label': label,
            'input_text': input_text,  # Keep original for reference
            'answer': answer
        }

    print("\nProcessing dataset with Phi-3.5 chat template...")
    processed_dataset = dataset.map(
        process_sft_example,
        desc="Formatting prompts"
    )

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE FORMATTED DATA (SFT):")
    print("=" * 80)
    print("[PROMPT]")
    print(processed_dataset['train'][0]['prompt'][:500] + "...")
    print("\n[COMPLETION]")
    print(processed_dataset['train'][0]['completion'])
    print("=" * 80)

    # Save processed dataset
    output_path = os.path.join(args.output_dir, 'sft')
    print(f"\nSaving processed dataset to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    processed_dataset.save_to_disk(output_path)

    print("\n✓ SFT preprocessing complete!")
    return processed_dataset


def preprocess_for_grpo(args):
    """Preprocess dataset for GRPO training with Phi-3.5."""
    print("=" * 80)
    print("Preprocessing Corr2Cause dataset for GRPO with Phi-3.5-Mini")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # Phi-3.5 requires pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load dataset
    print(f"\nLoading dataset from HuggingFace...")
    dataset = load_dataset(
        'causal-nlp/corr2cause',
        cache_dir=args.cache_dir
    )

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    # Process function
    def process_grpo_example(example):
        """Process single example for GRPO."""
        input_text = example['input']
        label = example['label']

        # Format query using Phi-3.5 chat template (no answer)
        query = format_grpo_query_phi3(tokenizer, input_text)

        return {
            'query': query,
            'label': label,
            'input_text': input_text  # Keep original for reference
        }

    print("\nProcessing dataset with Phi-3.5 chat template...")
    processed_dataset = dataset.map(
        process_grpo_example,
        desc="Formatting queries"
    )

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE FORMATTED QUERY (GRPO):")
    print("=" * 80)
    print(processed_dataset['train'][0]['query'][:500] + "...")
    print("=" * 80)

    # Save processed dataset
    output_path = os.path.join(args.output_dir, 'grpo')
    print(f"\nSaving processed dataset to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    processed_dataset.save_to_disk(output_path)

    print("\n✓ GRPO preprocessing complete!")
    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess Corr2Cause dataset for Phi-3.5-Mini")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sft', 'grpo', 'both'],
        default='both',
        help='Preprocessing mode: sft, grpo, or both'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/Phi-3.5-mini-instruct',
        help='Model name for tokenizer'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed',
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./data/cache',
        help='Cache directory for HuggingFace datasets'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run preprocessing
    if args.mode in ['sft', 'both']:
        preprocess_for_sft(args)

    if args.mode in ['grpo', 'both']:
        preprocess_for_grpo(args)

    print("\n" + "=" * 80)
    print("✓ All preprocessing complete!")
    print("=" * 80)
    print(f"\nProcessed datasets saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run SFT training: sbatch submit_sft_phi3.slurm")
    print("  2. After SFT completes, run GRPO: sbatch submit_grpo_phi3.slurm")


if __name__ == "__main__":
    main()
