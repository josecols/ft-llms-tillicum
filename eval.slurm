#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

module load conda
conda activate ft-llms

SHARED_FS=/gpfs/scrubbed/jcols/projects/ft-llms
PREDICTIONS_PATH="$SHARED_FS/outputs/plos_validation.csv"
DATASET_PATH="$SHARED_FS/data/plos_validation.parquet"

srun python scripts/eval.py \
  --predictions-path "$PREDICTIONS_PATH" \
  --dataset-path "$DATASET_PATH"