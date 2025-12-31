#!/bin/bash
# Script to deploy the RLcausality project to TU/e HPC

# HPC configuration
HPC_USER="20250638"
HPC_HOST="hpc.tue.nl"
HPC_DESTINATION="~/RLcausality"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying RLcausality project to HPC...${NC}"

# Create remote directory if it doesn't exist
echo -e "${YELLOW}Creating remote directory...${NC}"
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_DESTINATION}"

# Sync files using rsync (more efficient than scp)
echo -e "${YELLOW}Transferring files...${NC}"
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'data/cache/' \
    --exclude 'data/processed/' \
    --exclude 'models/' \
    --exclude 'wandb/' \
    --exclude '*.log' \
    ./ ${HPC_USER}@${HPC_HOST}:${HPC_DESTINATION}/

echo -e "${GREEN}Transfer complete!${NC}"
echo ""
echo "Next steps:"
echo "1. SSH into HPC: ssh ${HPC_USER}@${HPC_HOST}"
echo "2. Navigate to project: cd ${HPC_DESTINATION}"
echo "3. Load required modules (see README_HPC.md)"
echo "4. Install dependencies: pip install -r requirements.txt"
echo "5. Run preprocessing: python scripts/data_preprocessing.py"
echo "6. Submit training job: sbatch submit_sft.slurm"
