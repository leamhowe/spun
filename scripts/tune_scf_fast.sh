#!/bin/bash
#SBATCH --job-name=fast-scf-tune
#SBATCH --output=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%A_%a.out
#SBATCH --error=/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/logs/%x_%A_%a.err
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --cpus-per-task=6      # Request CPUs per GPU, not for the whole task
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00        # Shorter time per trial is more efficient
#SBATCH --mem=32GB
#SBATCH --array=1-100          # Launch 100 parallel jobs (1 trial per job)

# Load necessary modules
source ~/miniconda3/bin/activate torch

# --- WORKAROUND for NFS file locking ---
# Add a random sleep from 0 to 60 seconds to stagger job starts.
# This prevents all jobs from hitting the database file at the exact same time.
sleep $((RANDOM % 60))
# -----------------------------------------

echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID} after a short delay."

# Run the Python script for a single trial.
# The -u flag ensures that the output is unbuffered.
# The --trials argument is no longer needed here, as the array controls the count.
python -u /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts/tune_scf_fast.py \
    --study-name "scf_tuning_fast" \
    --server "jasmin" \
    --epochs 30 \
    --patience 7

echo "SLURM job array task ${SLURM_ARRAY_TASK_ID} finished."