from transformers import (HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from typing import Optional, Dict
import torch
import yaml
import os
import json
from peft import PeftModel
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(model_path: str, use_4bit: bool = False):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model_kwargs = {"trust_remote_code": True, "device_map": "auto", "attn_implementation": "sdpa"}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4") 
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft_model:
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        model_name = adapter_config.get("base_model_name_or_path")
        print("Detected PEFT model. Loading base model + adapter...")
        model=AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model=AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer

def extract_answer(text: str) -> Optional[bool]:
    text_lower = text.lower()
    if "therefore: yes" in text_lower or "answer: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower or "answer: no" in text_lower:
        return False
    return None

def evaluate_dataset(model,tokenizer,dataset,max_new_tokens,batch_size,max_samples,show_samples) -> Dict:
    beginning=800
    dataset = dataset.select(range(beginning, min(beginning+max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    predictions = []
    no_answer_count = 0
    samples_shown = 0

    print(f"Evaluating on {total} samples...")
    # Process in batches
    for i in tqdm(range(0, total, batch_size)):
        batch_end = min(i + batch_size, total)
        batch = dataset[i:batch_end]
        prompts = batch["query"]

        # Note: prompts are already formatted with chat template from preprocessing
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

        # Decode and extract answers
        prompt_len = inputs["input_ids"].shape[1]      # padded batch length
        gen_only_ids = outputs[:, prompt_len:]         # generated tokens only (works with LEFT padding)
        gen_texts = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        for j, generated_text in enumerate(gen_texts):
            predicted = extract_answer(generated_text)
            ground_truth = int(batch["label"][j]) == 1  # safer than bool(...)

            if predicted is None:
                no_answer_count += 1
            elif predicted == ground_truth:
                correct += 1

            predictions.append({
                "input": batch["query"][j],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "generated_text": generated_text,
                "is_correct": (predicted == ground_truth) if predicted is not None else False,
            })


            # Display sample predictions
            if samples_shown < show_samples:
                samples_shown += 1
                print(f"SAMPLE {samples_shown}/{show_samples}")
                print(f"\n[PROMPT]")
                print(prompts[j])
                print(f"\n[GENERATED ANSWER]")
                print(generated_text)
                print(f"\n[EXTRACTED PREDICTION]: {predicted}")
                print(f"[GROUND TRUTH]: {ground_truth}")
                print(f"[CORRECT]: {'✓' if predicted == ground_truth else '✗'}")

    accuracy = correct / total if total > 0 else 0
    answered_rate = (total - no_answer_count) / total if total > 0 else 0

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

    print(f"Acc={accuracy:.2%}, F1={f1:.2%}, P={precision:.2%}, R={recall:.2%} ({correct}/{total})")

    # Return results dictionary
    return {
        'predictions': predictions,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'answered_rate': answered_rate,
            'total_samples': total,
            'correct': correct,
            'no_answer_count': no_answer_count,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'evaluation_config.yaml')
    config_path = os.path.abspath(config_path)
    config = load_config(config_path)
    os.makedirs(config['output_dir'], exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(config['model_path'], config['use_4bit'])
    data_path = config['local_data_path']
    dataset = load_from_disk(data_path)['train']
    results = evaluate_dataset(model=model,tokenizer=tokenizer,dataset=dataset,max_new_tokens=config['max_new_tokens'],batch_size=config['batch_size'],
            max_samples=config['max_samples'], show_samples=config['show_samples'])


    # Save all results to JSON files
    output_dir = config['output_dir']

    # Save complete results (predictions + metrics)
    complete_results_file = os.path.join(output_dir, 'complete_results.json')
    with open(complete_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved complete results to {complete_results_file}")

    print(f"Evaluation complete! All outputs saved to {output_dir}")
    

if __name__ == "__main__":
    main()
