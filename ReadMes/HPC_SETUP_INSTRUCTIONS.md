# Quick HPC Setup Instructions

## What Went Wrong

Your SLURM job failed with these errors:
1. ❌ Virtual environment not found
2. ❌ Dependencies not installed (datasets, torch, etc.)
3. ❌ torchrun command not available

## The Fix

I've created an **automated setup script** that handles everything. Just follow these steps:

## Step-by-Step

### 1. Transfer Updated Files to HPC

From your local machine (Windows):

```bash
bash deploy_to_hpc.sh
# OR
deploy_to_hpc.bat
```

### 2. SSH to HPC

```bash
ssh 20250638@hpc.tue.nl
cd ~/RLcausality
```

### 3. Run Setup Script (ONE TIME ONLY)

⚠️ **IMPORTANT**: Run this directly on the login node, **NOT with sbatch**!

```bash
# Correct way:
bash setup_hpc.sh

# WRONG - don't do this:
# sbatch setup_hpc.sh  ❌
```

**This script will** (takes ~10-15 minutes):
- Load Python, CUDA, cuDNN modules
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies from requirements.txt
- Verify everything is working

**You only need to run this ONCE**. After it completes successfully, you're ready to train!

### 4. Submit Your Job

```bash
sbatch submit_sft.slurm
```

### 5. Monitor Progress

```bash
# Check if job is running
squeue -u 20250638

# Watch output live
tail -f logs/sft_*.out

# Check for errors
tail -f logs/sft_*.err
```

## What Changed

### New Files Created:
- **`setup_hpc.sh`** - Automated setup script (run once)
- **`scripts/verify_setup.py`** - Checks if environment is correct

### Updated Files:
- **`submit_sft.slurm`** - Now checks if venv exists before running
- **`submit_grpo.slurm`** - Now checks if venv exists before running
- **`README_HPC.md`** - Updated with detailed instructions

## Common Issues

### If setup_hpc.sh fails:

**Check Python module**:
```bash
module avail | grep Python
# Use whatever Python 3.10+ is available
```

**Check CUDA module**:
```bash
module avail | grep CUDA
# Use whatever CUDA 11.x is available
```

**Manually adjust modules** in `setup_hpc.sh` if needed (lines 14-16).

### If job still fails:

**Verify setup worked**:
```bash
source venv/bin/activate
python scripts/verify_setup.py
```

**Check torchrun is available**:
```bash
source venv/bin/activate
which torchrun  # Should show: ~/RLcausality/venv/bin/torchrun
```

## Success Indicators

✅ `setup_hpc.sh` completes without errors
✅ `verify_setup.py` shows all checks passed
✅ `which torchrun` shows the venv path
✅ SLURM job starts without "command not found" errors

## Next Steps After Successful Setup

1. **SFT Training** (~1-2 hours on 10k samples):
   ```bash
   sbatch submit_sft.slurm
   ```

2. **Monitor** with WandB at: https://wandb.ai

3. **After SFT completes**, run GRPO:
   ```bash
   sbatch submit_grpo.slurm
   ```

## Need Help?

Check the detailed guide: [README_HPC.md](README_HPC.md)

All troubleshooting solutions are in the "Troubleshooting" section.
