# Model Evaluation Guide

This guide explains how to evaluate your trained SFT and GRPO models on the Corr2Cause test dataset.

## Quick Start

To evaluate both models at once:

```bash
bash evaluate_both_models.sh
```

This will automatically:
1. Submit an evaluation job for the SFT model
2. Find the latest GRPO checkpoint and submit an evaluation job for it
3. Display the job IDs so you can monitor progress

## Manual Evaluation

### Evaluate SFT Model

```bash
sbatch submit_evaluation.slurm ./models/sft_model sft
```

### Evaluate GRPO Model

First, find the checkpoint you want to evaluate:

```bash
ls -d ./models/grpo_model/checkpoint-*
```

Then evaluate it:

```bash
sbatch submit_evaluation.slurm ./models/grpo_model/checkpoint-100 grpo
```

## Direct Python Execution (without SLURM)

If you want to run evaluation directly without SLURM:

```bash
# Activate environment
source venv/bin/activate

# Evaluate SFT model
python scripts/evaluate_model.py \
    --model_path ./models/sft_model \
    --model_type sft \
    --subsets default perturbation_by_refactorization perturbation_by_paraphrasing \
    --output_dir ./evaluation_results \
    --use_4bit

# Evaluate GRPO model
python scripts/evaluate_model.py \
    --model_path ./models/grpo_model/checkpoint-100 \
    --model_type grpo \
    --subsets default perturbation_by_refactorization perturbation_by_paraphrasing \
    --output_dir ./evaluation_results \
    --use_4bit
```

## Monitoring Jobs

Check your running jobs:

```bash
squeue -u $USER
```

View evaluation logs:

```bash
# Watch logs in real-time
tail -f logs/eval_<JOB_ID>.out

# Or just view the file
less logs/eval_<JOB_ID>.out
```

## Results

Results are saved in `evaluation_results/` with filenames like:
- `evaluation_sft_sft_model.json` - SFT model results
- `evaluation_grpo_checkpoint-100.json` - GRPO model results

Each results file contains:
- Overall accuracy on each test subset
- Number of correct predictions
- Number of samples where the model didn't provide a clear answer
- Answered rate (percentage of samples with clear yes/no answers)

### Example Results Format

```json
{
  "model_path": "./models/sft_model",
  "model_type": "sft",
  "results": {
    "default": {
      "total_samples": 500,
      "correct": 425,
      "accuracy": 0.85,
      "no_answer_count": 10,
      "answered_rate": 0.98
    },
    "perturbation_by_refactorization": {
      "total_samples": 200,
      "correct": 165,
      "accuracy": 0.825,
      "no_answer_count": 5,
      "answered_rate": 0.975
    },
    "perturbation_by_paraphrasing": {
      "total_samples": 200,
      "correct": 170,
      "accuracy": 0.85,
      "no_answer_count": 8,
      "answered_rate": 0.96
    }
  }
}
```

## Test Subsets Explained

1. **default**: Standard test set from Corr2Cause
2. **perturbation_by_refactorization**: Tests robustness to mathematically equivalent reformulations
3. **perturbation_by_paraphrasing**: Tests robustness to linguistic variations

## Troubleshooting

### Out of Memory Error

If you encounter OOM errors, you can:

1. Reduce batch size (already set to 1 in submit_evaluation.slurm)
2. Reduce max_new_tokens:
   ```bash
   python scripts/evaluate_model.py \
       --model_path ./models/sft_model \
       --model_type sft \
       --subsets default \
       --max_new_tokens 256 \
       --use_4bit
   ```

### No Checkpoints Found

If GRPO training is still running, wait for it to complete. Checkpoints are saved during training at intervals specified in the config (every 100 steps by default).

Check GRPO training status:
```bash
squeue -u $USER
tail -f logs/grpo_*.out
```

### Model Not Loading

Ensure the model path is correct and contains either:
- For SFT: `adapter_config.json` and adapter weights
- For GRPO: Same as above

Check the model directory:
```bash
ls -la ./models/sft_model/
ls -la ./models/grpo_model/checkpoint-100/
```
