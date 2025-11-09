#!/bin/bash
#SBATCH --job-name=ftllm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

module load conda
conda activate ft-llms

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -lc "hostname -I | awk '{print \$1}'")
echo "Head node: $head_node  IP: $head_ip"

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SHARED_FS=/gpfs/scrubbed/jcols/projects/ft-llms
CONFIG_PATH="$SHARED_FS/configs/llama3_2_3B_cuda.yaml"

srun tune run \
  --nnodes $SLURM_NNODES \
  --nproc_per_node $SLURM_GPUS_PER_TASK \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint "$head_ip:29500" \
  lora_finetune_distributed \
  --config "$CONFIG_PATH"