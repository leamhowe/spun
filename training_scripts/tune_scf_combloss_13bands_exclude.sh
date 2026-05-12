#!/bin/bash
#SBATCH --job-name=20251118-tune-scf-combloss-13bands-SCALE10000  # CHANGED: Updated job name
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
#SBATCH --exclude=gpuhost007,gpuhost016

# Load necessary modules
source ~/miniconda3/bin/activate torch

# --- WORKAROUND for NFS file locking ---
sleep $((RANDOM % 60))
# -----------------------------------------

echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID} for combloss tuning."

srun python -u /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/tune_scf_combloss_13bands.py \
    --study-name "20251118_tune_scf_combloss_13bands_SCALE10000_softcon" \
    --weights "sentinel2_softcon" \
    --server "jasmin" \
    --epochs 100 \
    --patience 20 \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds 
    # --use_indices \
    # --mask_clouds

echo "SLURM job array task ${SLURM_ARRAY_TASK_ID} finished."