#!/bin/bash
#SBATCH --job-name=20251118-tune-scf-combloss-13bands-SCALE10000  # CHANGED: Updated job name
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-100

# Load necessary modules
micromamba activate torch

# --- WORKAROUND for NFS file locking ---
sleep $((RANDOM % 60))
# -----------------------------------------

echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID} for combloss tuning."

srun python -u tune_scf_combloss_13bands.py \
    --study-name "tune_scf_combloss_13bands_SCALE10000_softcon" \
    --weights "sentinel2_softcon" \
    --server "jasmin" \
    --epochs 100 \
    --patience 20 \
    --split_file "<path/to/SPUN_Dataset>/dataset_metadata.csv" \
    --use_augmentations \
    --train_with_clouds 

echo "SLURM job array task ${SLURM_ARRAY_TASK_ID} finished."