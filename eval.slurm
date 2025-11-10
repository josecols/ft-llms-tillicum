#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

module load conda
conda activate ft-llms

SHARED_FS=$FT_LLMS_ROOTDIR
PREDICTIONS_PATH="$SHARED_FS/outputs/plos_validation.csv"
DATASET_PATH="$SHARED_FS/data/plos_validation.parquet"

srun python scripts/eval.py \
  --predictions-path "$PREDICTIONS_PATH" \
  --dataset-path "$DATASET_PATH"