# HPC Deployment Guide for TU/e

This guide provides specific instructions for deploying and running the LLM fine-tuning project on the TU/e HPC cluster.

## ⚡ Quick Start (Recommended)

### 1. Transfer Files to HPC

From your local Windows machine:

```bash
# Using Git Bash or WSL
bash deploy_to_hpc.sh

# Or using Windows Command Prompt
deploy_to_hpc.bat
```

This transfers all files to `~/RLcausality` on the HPC.

### 2. SSH into HPC

```bash
ssh 20250638@hpc.tue.nl
cd ~/RLcausality
```

### 3. Run Automated Setup

**This is the easiest method** - just run one script:

```bash
bash setup_hpc.sh
```

This script will:
- ✅ Load required modules (Python, CUDA, cuDNN)
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA support
- ✅ Install all dependencies from requirements.txt
- ✅ Create necessary directories
- ✅ Verify installation

**Time**: ~10-15 minutes

After setup completes, you're ready to submit jobs! Skip to step 5.

---

## 🔧 Manual Setup (Alternative)

If you prefer manual control or the automated setup fails:

### 3a. Load Required Modules

Check available modules:
```bash
module avail
```

Load Python, CUDA, and cuDNN:
```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
```

### 3b. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3c. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3d. Verify Installation

Run the verification script:

```bash
python scripts/verify_setup.py
```

This checks if all packages are installed correctly.

---

### 4. Configure WandB (Optional)

For training metrics logging:

```bash
source venv/bin/activate
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### 5. Submit Training Jobs

#### SFT Training

```bash
# Create logs directory
mkdir -p logs

# Submit SFT job
sbatch submit_sft.slurm
```

Monitor the job:
```bash
# Check job status
squeue -u 20250638

# View output (replace JOBID)
tail -f logs/sft_JOBID.out

# View errors
tail -f logs/sft_JOBID.err
```

#### GRPO Training (After SFT Completes)

```bash
# Submit GRPO job
sbatch submit_grpo.slurm
```

## SLURM Configuration

The SLURM scripts ([submit_sft.slurm](submit_sft.slurm) and [submit_grpo.slurm](submit_grpo.slurm)) are pre-configured with:

- **GPUs**: 4 GPUs per node
- **Time**: 24 hours (SFT), 48 hours (GRPO)
- **Memory**: 128GB
- **CPUs**: 16 cores

### Customizing SLURM Scripts

Edit the `#SBATCH` directives in the `.slurm` files:

```bash
#SBATCH --gres=gpu:4              # Number of GPUs (1, 2, 4, 8)
#SBATCH --time=24:00:00           # Max runtime (HH:MM:SS)
#SBATCH --mem=128G                # Memory
#SBATCH --cpus-per-task=16        # CPU cores
#SBATCH --partition=gpu           # Partition name (check with 'sinfo')
```

Check available partitions and GPUs:
```bash
sinfo
squeue
```

## Useful SLURM Commands

```bash
# Submit job
sbatch submit_sft.slurm

# Check job status
squeue -u 20250638

# Check specific job
squeue -j JOBID

# Cancel job
scancel JOBID

# Cancel all your jobs
scancel -u 20250638

# View job details
scontrol show job JOBID

# View past jobs
sacct -u 20250638

# View GPU usage during job
ssh <node_name>  # Get node name from squeue
nvidia-smi
```

## Model Checkpoints

Checkpoints are saved to:
- SFT: `./models/sft_model/`
- GRPO: `./models/grpo_model/`

To download checkpoints to your local machine:

```bash
# From your local machine
rsync -avz --progress 20250638@hpc.tue.nl:~/RLcausality/models/ ./models/
```

## Monitoring Training

### WandB (Recommended)

If you configured WandB, view training progress at:
https://wandb.ai/your-username/corr2cause-sft

### TensorBoard (Alternative)

If using TensorBoard instead:

```bash
# On HPC, start TensorBoard (in a separate session)
tensorboard --logdir ./models/sft_model --host 0.0.0.0 --port 6006

# On local machine, create SSH tunnel
ssh -L 6006:localhost:6006 20250638@hpc.tue.nl

# Open in browser
http://localhost:6006
```

### Log Files

Training logs are in `logs/`:
```bash
# Follow SFT training
tail -f logs/sft_JOBID.out

# Follow GRPO training
tail -f logs/grpo_JOBID.out

# Check errors
tail -f logs/sft_JOBID.err
```

## Troubleshooting

### Setup Issues

#### Virtual Environment Not Found

**Error**: `venv/bin/activate: No such file or directory`

**Solution**:
```bash
# Run the setup script
bash setup_hpc.sh

# Or create manually
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### ModuleNotFoundError: No module named 'datasets'

**Error**: `ModuleNotFoundError: No module named 'datasets'` (or torch, transformers, etc.)

**Solution**:
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### torchrun: command not found

**Error**: `/var/spool/slurmd/job*/slurm_script: line XX: torchrun: command not found`

**Solution**:
```bash
# PyTorch not installed correctly. Reinstall:
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify with:
```bash
which torchrun  # Should show: ~/RLcausality/venv/bin/torchrun
```

### Module Not Found

If modules are not available, contact HPC support or check documentation:
```bash
module spider Python
module spider CUDA
```

### CUDA Out of Memory

1. Edit `configs/sft_config.yaml` or `configs/grpo_config.yaml`
2. Reduce `per_device_train_batch_size`
3. Increase `gradient_accumulation_steps`
4. Enable `use_4bit: true` for quantization
5. Request more GPUs in SLURM script

### Job Pending for Long Time

Check cluster load:
```bash
squeue
sinfo
```

Request fewer resources or different partition if needed.

### Permission Denied

Ensure scripts are executable:
```bash
chmod +x deploy_to_hpc.sh
chmod +x run_sft.sh
chmod +x run_grpo.sh
```

## Resource Estimation

### Storage Requirements

- Dataset cache: ~5GB
- Processed data: ~10GB
- SFT model (7B with LoRA): ~15GB
- GRPO model: ~15GB
- Logs and checkpoints: ~50GB

**Total**: ~100GB recommended

Check your quota:
```bash
quota -s
df -h ~
```

### Training Time Estimates (4x A100 GPUs)

- **SFT (7B model, 3 epochs)**: 8-12 hours
- **GRPO (7B model, 1 epoch)**: 20-30 hours

Adjust `--time` in SLURM scripts accordingly.

## Best Practices

1. **Test First**: Run a small test job before full training
   ```bash
   # Edit config to use fewer steps
   max_steps: 100
   ```

2. **Save Checkpoints Frequently**: Set in config files
   ```yaml
   save_steps: 100
   save_total_limit: 5
   ```

3. **Monitor GPU Usage**:
   ```bash
   # SSH to compute node during job
   nvidia-smi -l 1
   ```

4. **Use Screen/Tmux** for long-running commands:
   ```bash
   # Start screen session
   screen -S training

   # Detach: Ctrl+A, D
   # Reattach: screen -r training
   ```

5. **Backup Important Checkpoints**:
   ```bash
   # Copy to backup location
   cp -r models/sft_model /backup/path/
   ```

## Getting Help

- **TU/e HPC Documentation**: Check internal HPC wiki
- **SLURM Documentation**: https://slurm.schedmd.com/
- **TRL Documentation**: https://huggingface.co/docs/trl/
- **HPC Support**: Contact your HPC support team

## Clean Up

After training completes, clean up temporary files:

```bash
# Remove cache
rm -rf data/cache/

# Remove old checkpoints (keep only final models)
rm -rf models/sft_model/checkpoint-*
rm -rf models/grpo_model/checkpoint-*

# Clean wandb logs (if not needed)
rm -rf wandb/
```
