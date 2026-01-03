"""
Prepare SFT (Supervised Fine-Tuning) dataset from GPT evaluation results.
This script filters correct reasoning traces and prepares them for fine-tuning.
"""
import json
import os
from datasets import Dataset, DatasetDict

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
        formatted_data.append({
            'query': trace['input'],
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
    predictions_file = './evaluation_results/gpt_predictions_gpt-5.2_20260101_195905.json'
    output_dir = './data/processed/sft_deepseek'

    os.makedirs(output_dir, exist_ok=True)
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    correct_traces = filter_correct_traces(predictions_data)

    formatted_data = format_for_sft(correct_traces)

    print("\nCreating train/val split...")
    train_data, val_data = create_train_val_split(formatted_data, val_ratio=0.15)

    # Create HuggingFace Dataset
    print("\nCreating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # Save to disk
    print(f"\nSaving datasets to {output_dir}...")
    dataset_dict.save_to_disk(output_dir)

if __name__ == "__main__":
    main()
