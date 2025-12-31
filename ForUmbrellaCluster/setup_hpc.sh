#!/bin/bash
# One-time setup script for TU/e HPC
# Run this ONCE after transferring files to HPC
#
# IMPORTANT: Run this directly on the LOGIN NODE, NOT via sbatch!
#   Correct:   bash setup_hpc.sh
#   Wrong:     sbatch setup_hpc.sh
#

set -e  # Exit on any error

echo "=================================="
echo "RLcausality HPC Setup Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found!${NC}"
    echo "Please run this script from the RLcausality directory"
    exit 1
fi

echo -e "${YELLOW}Step 1: Loading required modules...${NC}"
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

echo -e "${GREEN}✓ Modules loaded${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Step 2: Checking Python version...${NC}"
python --version
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Python not available${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python available${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}Step 3: Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python -m venv venv
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
    echo "Try: python3 -m venv venv"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Step 4: Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}Step 5: Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install PyTorch first (with CUDA support)
echo -e "${YELLOW}Step 6: Installing PyTorch with CUDA support...${NC}"
echo "This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install PyTorch${NC}"
    exit 1
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install other requirements
echo -e "${YELLOW}Step 7: Installing other dependencies from requirements.txt...${NC}"
echo "This may take several minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install requirements${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}Step 8: Creating directories...${NC}"
mkdir -p data/cache
mkdir -p data/processed
mkdir -p models
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Verify installation
echo -e "${YELLOW}Step 9: Verifying installation...${NC}"
python scripts/verify_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}WARNING: Verification found some issues${NC}"
    echo "You may still be able to proceed, but check the errors above"
else
    echo -e "${GREEN}✓ Verification passed${NC}"
fi
echo ""

echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Preprocess data: sbatch submit_preprocessing.slurm"
echo "   OR interactively: source venv/bin/activate && python scripts/data_preprocessing.py"
echo ""
echo "2. Submit SFT training: sbatch submit_sft.slurm"
echo ""
echo "3. After SFT completes, submit GRPO: sbatch submit_grpo.slurm"
echo ""
echo "To check your jobs: squeue -u $USER"
echo ""
