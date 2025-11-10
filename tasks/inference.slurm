#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

module load conda
conda activate ft-llms

SHARED_FS="$FT_LLMS_ROOT"
ADAPTER_PATH="$SHARED_FS/checkpoints/llama3_2_3B_lora/epoch_0/"
OUTPUT_PATH="$SHARED_FS/outputs/plos_validation"
DATASET_PATH="$SHARED_FS/data/plos_validation.parquet"

srun python scripts/inference.py \
  --adapter-path "$ADAPTER_PATH" \
  --output-path "$OUTPUT_PATH" \
  --dataset-path "$DATASET_PATH" \
  --batch-size 16 \
  --max-new-tokens 256