"""
Prepare SFT (Supervised Fine-Tuning) dataset from GPT evaluation results.
This script filters correct reasoning traces and prepares them for fine-tuning.
"""
import json
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Initialize tokenizer for chat template formatting
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)

def load_gpt_predictions(predictions_file: str):
    """Load GPT predictions from JSON file."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def filter_correct_traces(predictions_data):
    predictions = predictions_data['predictions']
    correct_traces = [p for p in predictions if p['is_correct']]
    print(f"Total samples: {len(predictions)}")
    print(f"Correct samples: {len(correct_traces)}")
    return correct_traces

def format_for_sft(correct_traces):
    formatted_data = []
    for trace in correct_traces:
        # Apply chat template to the input query
        # trace['input'] contains the raw user prompt
        messages = [{"role": "user", "content": trace['input']}]
        formatted_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        formatted_data.append({
            'query': formatted_query,  # Now properly formatted with chat template
            'response': trace['generated_text'],
            'label': trace['ground_truth'],
            'predicted': trace['predicted']
        })
    return formatted_data

def create_train_val_split(data, val_ratio=0.15):
    """Split data into train and validation sets."""
    import random
    random.seed(42)

    indices = list(range(len(data)))
    random.shuffle(indices)

    val_size = int(len(data) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return train_data, val_data

def main():
    # Paths
    evaluation_results_dir = './evaluation_results'
    output_dir = './data/processed/sft_deepseek'

    os.makedirs(output_dir, exist_ok=True)

    # Find all GPT prediction JSON files in the evaluation_results directory
    import glob
    prediction_files = glob.glob(os.path.join(evaluation_results_dir, 'gpt_predictions_*.json'))

    print(f"Found {len(prediction_files)} prediction files:")

    # Load and merge all correct traces from all files
    all_correct_traces = []
    for predictions_file in prediction_files:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        correct_traces = filter_correct_traces(predictions_data)
        all_correct_traces.extend(correct_traces)

    print(f"\n{'='*60}")
    print(f"Total correct traces from all files: {len(all_correct_traces)}")
    print(f"{'='*60}")

    # Format the merged data for SFT
    formatted_data = format_for_sft(all_correct_traces)

    # Create HuggingFace Dataset
    print("\nCreating HuggingFace dataset...")
    train_dataset = Dataset.from_list(formatted_data)
    print(f"Train samples: {len(train_dataset)}")

    dataset_dict = DatasetDict({
        'train': train_dataset
    })

    # Save to disk
    print(f"\nSaving datasets to {output_dir}...")
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved successfully!")

if __name__ == "__main__":
    main()
