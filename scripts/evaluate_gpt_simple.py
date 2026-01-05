from openai import OpenAI
from typing import Optional, Dict
import json
import os
from datasets import load_from_disk
from tqdm import tqdm
from datetime import datetime

def extract_answer(text: str) -> Optional[bool]:
    """Extract yes/no answer from generated text."""
    text_lower = text.lower()
    if "therefore: yes" in text_lower or "answer: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower or "answer: no" in text_lower:
        return False
    return None

def evaluate_gpt_on_dataset(
    client: OpenAI,
    model_name: str,
    dataset,
    max_samples,
    output_file: str,
    reasoning_effort: str,
    show_samples: int = 3,
    beginning: int = 0,
) -> Dict:
    
    dataset = dataset.select(range(beginning, min(max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    predictions = []
    no_answer_count = 0
    samples_shown = 0
    api_errors = 0
    empty_content_count = 0
    length_cutoff_count = 0

    print(f"Evaluating {model_name} on {total} samples...")

    # Process one sample at a time
    for i in tqdm(range(total)):
        sample = dataset[i]
        prompt = sample["query"]
        ground_truth = bool(sample['label'])

        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
                reasoning_effort=reasoning_effort
            )

            # Extract response metadata
            finish_reason = response.choices[0].finish_reason
            generated_text = response.choices[0].message.content

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n⚠️  WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                print(f"   Full response choice: {response.choices[0]}")
                print(f"   Usage: {response.usage}")
                generated_text = generated_text or ""  # Convert None to empty string

            # Track token limit cutoffs
            if finish_reason == "length":
                length_cutoff_count += 1
                print(f"\n✂️  Sample {i}: Response cut off due to token limit")
                print(f"   Tokens used: {response.usage.completion_tokens}/{response.usage.total_tokens}")

            predicted = extract_answer(generated_text)

            if predicted is None:
                no_answer_count += 1
            elif predicted == ground_truth:
                correct += 1

            predictions.append({
                'index': i,
                'input': prompt,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'generated_text': generated_text,
                'is_correct': predicted == ground_truth if predicted is not None else False,
                'num_variables': sample.get('num_variables', None),
                'template': sample.get('template', None),
                'finish_reason': finish_reason,
                'token_usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            })

            # Display sample predictions
            if samples_shown < show_samples:
                samples_shown += 1
                print(f"\n{'='*80}")
                print(f"SAMPLE {samples_shown}/{show_samples}")
                print(f"{'='*80}")
                print(f"\n[PROMPT]")
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print(f"\n[GENERATED ANSWER]")
                print(generated_text)
                print(f"\n[EXTRACTED PREDICTION]: {predicted}")
                print(f"[GROUND TRUTH]: {ground_truth}")
                print(f"[CORRECT]: {'✓' if predicted == ground_truth else '✗'}")
                print(f"{'='*80}\n")

        except Exception as e:
            api_errors += 1
            print(f"\nError on sample {i}: {str(e)}")
            predictions.append({
                'index': i,
                'input': prompt,
                'ground_truth': ground_truth,
                'predicted': None,
                'generated_text': f"ERROR: {str(e)}",
                'is_correct': False,
                'num_variables': sample.get('num_variables', None),
                'template': sample.get('template', None),
                'error': str(e)
            })

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    answered_rate = (total - no_answer_count - api_errors) / total if total > 0 else 0

    true_positives = sum(1 for p in predictions if p['predicted'] is True and p['ground_truth'] is True)
    false_positives = sum(1 for p in predictions if p['predicted'] is True and p['ground_truth'] is False)
    false_negatives = sum(1 for p in predictions if p['predicted'] is False and p['ground_truth'] is True)
    true_negatives = sum(1 for p in predictions if p['predicted'] is False and p['ground_truth'] is False)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'model': model_name,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'answered_rate': answered_rate,
        'no_answer_count': no_answer_count,
        'api_errors': api_errors,
        'empty_content_count': empty_content_count,
        'length_cutoff_count': length_cutoff_count,
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {model_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"F1 Score: {f1:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Answered Rate: {answered_rate:.2%}")
    print(f"No Answer Count: {no_answer_count}")
    print(f"API Errors: {api_errors}")
    print(f"Empty Content Responses: {empty_content_count}")
    print(f"Token Limit Cutoffs: {length_cutoff_count}")
    print(f"{'='*80}\n")

    # Save all predictions if output file is specified
    
    output_data = {
            'metadata': results,
            'predictions': predictions
        }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    print(f"All predictions saved to: {output_file}")

    return results


def main():
    # Configuration
    model_name = "gpt-5.2"  # Change to "gpt-4", "gpt-4-turbo", etc. as needed
    max_samples = 200
    show_samples = 3
    reasoning_effort = "medium"  # Options: "low", "medium" (default), "high"
    beginning = 100

    # Data path - using GPT formatted data
    data_path = "./data/processed/GPT"

    # Output file for saving all predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./evaluation_results"
    output_file = os.path.join(output_dir, f"gpt_predictions_{model_name}_{timestamp}.json")

    # Initialize OpenAI client
    # Make sure you have OPENAI_API_KEY set in your environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    client = OpenAI(api_key=api_key)

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_from_disk(data_path)['train']
    print(f"Dataset loaded: {len(dataset)} samples in data set")

    # Evaluate
    results = evaluate_gpt_on_dataset(
        client=client,
        model_name=model_name,
        dataset=dataset,
        max_samples=max_samples,
        show_samples=show_samples,
        output_file=output_file,
        reasoning_effort=reasoning_effort,
        beginning=beginning
    )

    print("Evaluation complete!")
    print(f"Results summary saved to: {output_file}")


if __name__ == "__main__":
    main()
