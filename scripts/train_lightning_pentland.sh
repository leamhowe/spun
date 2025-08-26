#!/bin/bash
#SBATCH --job-name=unet_torchgeolt
# #SBATCH --job-name=$model_name
#SBATCH --output=/home/s2112771/r/1_DETECT/unet/output/%x_%j.out
#SBATCH --error=/home/s2112771/r/1_DETECT/unet/output/%x_%j.err
#SBATCH --partition gpgpu
#SBATCH --account geos_gpgpu
#SBATCH --gres=gpu:t4:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


# Load necessary modules
# module load python/3.8
# module load cuda/11
# module load pytorch/1.7.1

# Activate virtual environment
# source ~/miniforge3/bin/activate lightning
# source ~/miniconda3/bin/activate torchgeeo
source ~/r/miniforge3/bin/activate torchgeeo
# mamba activate lightning

# Navigate to the directory containing your script
cd /home/s2112771/r/1_DETECT/unet/scripts

# Model name argument: default if not provided
model_name=${1:-default_model_name}

# Path to Python script
script_name="train_lightning.py"

# Run your Python script with model name as an argument
# srun python $script_name --model_name $model_name --use_augmentations --bands wang_bands
# srun python $script_name --model_name $model_name --use_augmentations --bands all_bands
# srun python $script_name --model_name $model_name --use_augmentations --bands all_spectral_bands
# srun python $script_name --model_name $model_name --use_augmentations --bands custom_bands
# srun python -u $script_name --model_name $model_name --bands custom_bands  --use_augmentations
srun python -u $script_name --model_name $model_name --bands custom_bands --server pentland
# srun python -u $script_name --model_name $model_name --bands all_spectral_bands  
# srun python -u $script_name --model_name $model_name --bands wang_bands  
# srun python -u $script_name --model_name $model_name --bands all_bands  
# srun python -u $script_name --model_name $model_name --bands perm_threshold_01
python -u $script_name --model_name $model_name --bands all_spectral_bands --server pentland --weights ResNet50_Weights.SENTINEL2_ALL_DINO --use_augmentations


model_name="20250606_NewBinaryDataset_TrainCairngorm_ValGlencoe_MaskClouds_unet_torchgeolt_BBresnet50_BAllSpectral_DiceMacro_pat100_nonorm"
# python -u train_lightning.py --model_name 20250606_NewBinaryDataset_TrainCairngorm_ValGlencoe_MaskClouds_unet_torchgeolt_BBresnet50_B0203040911_DiceMacro_pat100_nonorm --bands custom_bands --server pentland --backbone resnet50 --mask_clouds 
python -u train_lightning.py --model_name 20250620_NewBinaryDatasetAug_TrainCairngorm_ValGlencoe_MaskClouds_unet_torchgeolt_BBresnet50_BAllNDSIcscdf_DiceMacro_nonorm --dataset cs_threshold_50_binary_local_mean_threshold_00 --bands custom_bands --server pentland --backbone resnet50 --mask_clouds --use_augmentations

# # Extracts optional arguments if provided
# # model_name=${1:-default_model_name}
# use_augmentations=${2:-0} # Based on usage, 0 means don't use, 1 means use
# bands=${3:-all_spectral_bands}
# backbone=${4:-resnet50}
# weights=${5:-SENTINEL2_ALL_DINO}
# lr=${6:-0.001}
# batch_size=${7:-16}
# patience=${8:-50}

# # Construct the command line with conditional use_augmentations
# cmd="python train.py --model_name $model_name --backbone $backbone --weights $weights --lr $lr --batch_size $batch_size --patience $patience --bands $bands"
# if [ "$use_augmentations" -eq 1 ]; then
#   cmd="$cmd --use_augmentations"
# fi

# # Run the Python script with optional parameters
# echo "Running command: $cmd"
# srun $cmd