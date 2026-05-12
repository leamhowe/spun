#!/bin/bash
#SBATCH --job-name=spun_training
#SBATCH --output=../training/slurm_logs/%x_%j.out
#SBATCH --error=../training/slurm_logs/%x_%j.err
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=orchid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

# Activate virtual environment
micromamba activate torch

# Navigate to the directory where the job was submitted (training_scripts)
cd $SLURM_SUBMIT_DIR

# Ensure output directory for SLURM logs exists
mkdir -p ../training/slurm_logs

# Model name argument: defaults to a timestamped name if not provided
model_name=${1:-"SPUN_run_$(date +%Y%m%d_%H%M)"}

echo "=========================================================="
echo "Starting SPUN Training Job: $model_name"
echo "Running on node: $HOSTNAME"
echo "=========================================================="

# Run the Python script with the optimal hyperparameter configuration
srun python -u train.py \
    --model_name "$model_name" \
    --data_root "/home/users/leamhowe/sensecdt/1_DETECT/data/spun_dataset_zenodo/SPUN_Dataset" \
    --split_file "dataset_metadata.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --weights "sentinel2_all_dino" \
    --loss_fn "huber" \
    --loss_weighting "none" \
    --dice_loss_weight 0.166421 \
    --snow_thresh 0.399728 \
    --lr 0.000118 \
    --batch_size 16 \
    --num_workers 24 \
    --patience 50

echo "=========================================================="
echo "Job Completed."
echo "=========================================================="