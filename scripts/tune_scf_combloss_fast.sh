#!/bin/bash
#SBATCH --job-name=tune-scf-combloss-fast  # CHANGED: Updated job name
#SBATCH --output=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%A_%a.out
#SBATCH --error=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%A_%a.err
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-100

# Load necessary modules
source ~/miniconda3/bin/activate torch

# --- WORKAROUND for NFS file locking ---
sleep $((RANDOM % 60))
# -----------------------------------------

echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID} for combloss tuning."

# --- CHANGED: Updated Python command ---
python -u /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/tune_scf_combloss_fast.py \
    --study-name "20251023_tuning_scf_combloss_final" \
    --server "jasmin" \
    --epochs 75 \
    --patience 15 \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds 
    # --- NEW: Add required split_file argument ---
    
    # --- NEW: Add fixed arguments for the study (optional, add/remove as needed) ---
    # --use_augmentations \
    # --use_indices \
    # --mask_clouds
    # --train_with_clouds
    # -----------------------------------------------------------------------------

echo "SLURM job array task ${SLURM_ARRAY_TASK_ID} finished."