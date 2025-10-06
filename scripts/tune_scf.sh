#!/bin/bash
#SBATCH --job-name=tune-scf-unet
#SBATCH --output=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%j.out
#SBATCH --error=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%j.err
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# Load necessary modules
source ~/miniconda3/bin/activate torch

# Run the Python script for tuning
# The -u flag ensures that the output is unbuffered and written directly to the log files.
python -u /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/tune_scf.py \
    --trials 100 \
    --study-name "scf_tuning_final" \
    --server "jasmin" \
    --epochs 50 \
    --patience 10 \
    --k-folds 3 \
    # --epochs 75 \
    # --patience 15 \
    # --k-folds 5    
    # --mask_clouds \
    # --use_indices
