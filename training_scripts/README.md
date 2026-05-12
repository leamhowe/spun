## 🏋️ Training the Model

Training is handled by `train.py`. The pipeline utilises Weights & Biases (`wandb`) for logging and Albumentations for data augmentation.

> [!NOTE]
> Training will be very slow if you do not have a GPU available. The code is optimised to take advantage of all GPUs available where the code is run.

### The Dataset

The dataset used to train the SPUN (Snow Patch U-Net) model contains **2,455 Sentinel-2 image chips** of patchy and late-season snow derived from 25 cm aerial surveys (GetMapping) of Scotland. 

To train the model from scratch you need to download the dataset. More information and the full dataset can be found here: https://doi.org/10.5281/zenodo.20055817 


### Optimal Run Command

Based on our extensive hyperparameter tuning and ablation studies, the optimal model utilises all 13 Sentinel-2 bands, a ResNet-50 backbone pre-trained with DINO, and a combined Huber + Dice loss function to balance continuous fraction prediction with accurate segmentation.

Run the optimal configuration locally or via SLURM:

```bash
python -m scripts.train \
    --model_name "Final_T15_dino_none_scale10000" \
    --data_root "<path_to_zenodo_dataset>/SPUN_Dataset" \
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

```

*Note: On JASMIN, you can submit this job using the provided SLURM wrapper: `sbatch scripts/train_lightning_jasmin.sh <run_name>*`