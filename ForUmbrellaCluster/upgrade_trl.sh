#!/bin/bash
# Script to upgrade TRL and related packages on HPC
# Run this on the LOGIN NODE, NOT via sbatch
#
# Usage: bash upgrade_trl.sh

set -e  # Exit on any error

echo "========================================="
echo "Upgrading TRL and Related Packages"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -d "venv" ]; then
    echo -e "${RED}ERROR: Virtual environment not found!${NC}"
    echo "Please run setup_hpc.sh first, or run this script from the RLcausality directory"
    exit 1
fi

# Load required modules
echo -e "${YELLOW}Loading Python modules...${NC}"
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
echo -e "${GREEN}✓ Modules loaded${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Show current versions
echo -e "${YELLOW}Current versions:${NC}"
pip show trl | grep Version
pip show transformers | grep Version
pip show peft | grep Version
echo ""

# Upgrade packages
echo -e "${YELLOW}Upgrading TRL, transformers, and peft...${NC}"
echo "This may take a few minutes..."
pip install --upgrade trl transformers peft accelerate

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to upgrade packages${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Packages upgraded${NC}"
echo ""

# Show new versions
echo -e "${YELLOW}New versions:${NC}"
pip show trl | grep Version
pip show transformers | grep Version
pip show peft | grep Version
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "
from trl import SFTTrainer
import transformers
import peft
print('✓ TRL imports successfully')
print('✓ Transformers version:', transformers.__version__)
print('✓ PEFT version:', peft.__version__)
import trl
print('✓ TRL version:', trl.__version__)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}WARNING: Verification had some issues${NC}"
    echo "You may still be able to proceed, but check the errors above"
else
    echo -e "${GREEN}✓ Verification passed${NC}"
fi
echo ""

echo "========================================="
echo -e "${GREEN}Upgrade Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Submit SFT training: sbatch submit_sft.slurm"
echo "2. Monitor progress: tail -f logs/sft_*.out"
echo ""