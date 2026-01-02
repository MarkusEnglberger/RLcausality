"""
Evaluation script for SFT and GRPO models on Corr2Cause test datasets.
Supports multiple test subsets: default, perturbation_by_refactorization, perturbation_by_paraphrasing
"""

import os
import re
import torch
import yaml
from typing import Optional, Dict, List
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_answer(text: str) -> Optional[bool]:
    """Extract the answer from generated text."""
    text_lower = text.lower()

    # Look for explicit answer patterns
    if "therefore: yes" in text_lower or "answer: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower or "answer: no" in text_lower:
        return False

    # Fallback: look for yes/no in the last part of the text
    last_lines = text_lower.split('\n')[-3:]  # Check last 3 lines
    last_text = ' '.join(last_lines)

    if re.search(r'\byes\b', last_text) and not re.search(r'\bno\b', last_text):
        return True
    elif re.search(r'\bno\b', last_text) and not re.search(r'\byes\b', last_text):
        return False

    return None



def load_model_and_tokenizer(model_path: str, use_4bit: bool = False):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading model from {model_path}...")

    # Get HuggingFace token from environment if available (needed for gated models)
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print("Using HuggingFace token from HF_TOKEN environment variable")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        token=hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if it's a PEFT model
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if is_peft_model:
        print("Detected PEFT model. Loading base model + adapter...")
        import json
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=hf_token,
            **model_kwargs
        )

        # Load adapter
        model = PeftModel.from_pretrained(model, model_path)
        print("Loaded PEFT adapter successfully")
    else:
        print("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            **model_kwargs
        )

    model.eval()
    return model, tokenizer


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int = 2048,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    show_samples: int = 3
) -> Dict:
    """Evaluate model on a dataset.

    Args:
        show_samples: Number of example predictions to display during evaluation
    """

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    predictions = []
    no_answer_count = 0
    samples_shown = 0

    print(f"Evaluating on {total} samples...")

    # Check which field name to use
    # - Preprocessed GRPO data uses 'query' (already fully formatted)
    # - Raw HuggingFace data uses 'input' (needs formatting)
    sample = dataset[0]
    print(f"Available fields in dataset: {list(sample.keys())}")

    # Process in batches
    for i in tqdm(range(0, total, batch_size)):
        batch_end = min(i + batch_size, total)
        batch = dataset[i:batch_end]

        # Handle both single sample and batch
        if isinstance(batch["query"], str):
            batch = {k: [v] for k, v in batch.items()}
        prompts = batch["query"]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(  **inputs,  max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract answers
        for j, output in enumerate(outputs):
            # Decode only the generated part (not the prompt)
            generated_text = tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Extract answer
            predicted = extract_answer(generated_text)
            ground_truth = bool(batch['label'][j])

            if predicted is None:
                no_answer_count += 1
            elif predicted == ground_truth:
                correct += 1

            predictions.append({
                'input': batch["query"][j],
                'ground_truth': ground_truth,
                'predicted': predicted,
                'generated_text': generated_text,
                'is_correct': predicted == ground_truth if predicted is not None else False
            })

            # Display sample predictions
            if samples_shown < show_samples:
                samples_shown += 1
                print(f"\n{'='*80}")
                print(f"SAMPLE {samples_shown}/{show_samples}")
                print(f"{'='*80}")
                print(f"\n[PROMPT]")
                print(prompts[j])
                print(f"\n[GENERATED ANSWER]")
                print(generated_text)
                print(f"\n[EXTRACTED PREDICTION]: {predicted}")
                print(f"[GROUND TRUTH]: {ground_truth}")
                print(f"[CORRECT]: {'✓' if predicted == ground_truth else '✗'}")
                print(f"{'='*80}\n")

    accuracy = correct / total if total > 0 else 0
    answered_rate = (total - no_answer_count) / total if total > 0 else 0

    # Calculate F1, Precision, Recall
    # True Positives: predicted True and ground truth is True
    # False Positives: predicted True but ground truth is False
    # False Negatives: predicted False but ground truth is True
    # True Negatives: predicted False and ground truth is False
    true_positives = sum(1 for p in predictions if p['predicted'] is True and p['ground_truth'] is True)
    false_positives = sum(1 for p in predictions if p['predicted'] is True and p['ground_truth'] is False)
    false_negatives = sum(1 for p in predictions if p['predicted'] is False and p['ground_truth'] is True)
    true_negatives = sum(1 for p in predictions if p['predicted'] is False and p['ground_truth'] is False)

    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'no_answer_count': no_answer_count,
        'answered_rate': answered_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'predictions': predictions
    }

    return results


def main():
    # Load config from default location
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'evaluation_config.yaml')
    config_path = os.path.abspath(config_path)

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Print configuration
    print("\n" + "="*80)
    print("Evaluation Configuration:")
    print("="*80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['model_path'], config['use_4bit'])

    # Evaluate on each subset
    all_results = {}

    for subset in config['subsets']:
        print(f"\n{'='*80}")
        print(f"Evaluating on subset: {subset}")
        print(f"{'='*80}\n")

        data_path = config['local_data_path']
        print(f"Loading local preprocessed test data from: {data_path}")
        # Load only the test split (HuggingFace datasets loads splits lazily)
        dataset = load_from_disk(data_path)['test']
        print(f"Loaded test split with {len(dataset)} samples")

        # Evaluate
        results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens=config['max_new_tokens'],
            batch_size=config['batch_size'],
            max_samples=config['max_samples'],
            show_samples=config['show_samples']
        )

        all_results[subset] = results

        # Print results
        print(f"\nResults for {subset}:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Correct: {results['correct']}")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  Recall: {results['recall']:.2%}")
        print(f"  F1 Score: {results['f1']:.2%}")
        print(f"  No answer count: {results['no_answer_count']}")
        print(f"  Answered rate: {results['answered_rate']:.2%}")
        print(f"  Confusion Matrix: TP={results['true_positives']}, FP={results['false_positives']}, FN={results['false_negatives']}, TN={results['true_negatives']}")

    # Save results
    model_name = os.path.basename(config['model_path'].rstrip('/'))
    output_file = os.path.join(
        config['output_dir'],
        f"evaluation_{model_name}.json"
    )

    # Save detailed results (without full predictions to keep file size manageable)
    summary_results = {
        'model_path': config['model_path'],
        'config': config,
        'results': {
            subset: {
                'total_samples': res['total_samples'],
                'correct': res['correct'],
                'accuracy': res['accuracy'],
                'precision': res['precision'],
                'recall': res['recall'],
                'f1': res['f1'],
                'no_answer_count': res['no_answer_count'],
                'answered_rate': res['answered_rate'],
                'confusion_matrix': {
                    'true_positives': res['true_positives'],
                    'false_positives': res['false_positives'],
                    'false_negatives': res['false_negatives'],
                    'true_negatives': res['true_negatives']
                }
            }
            for subset, res in all_results.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print overall summary
    print("Overall Summary:")
    for subset, results in all_results.items():
        print(f"  {subset}: Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}, P={results['precision']:.2%}, R={results['recall']:.2%} ({results['correct']}/{results['total_samples']})")


if __name__ == "__main__":
    main()