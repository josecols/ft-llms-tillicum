#!/bin/bash
#SBATCH --job-name=ftllm
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -lc "hostname -I | awk '{print \$1}'")
echo "Head node: $head_node  IP: $head_node_ip"

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO

SHARED_FS=/gpfs/scrubbed/jcols/projects/ft-llms
CONFIG_PATH="$SHARED_FS/configs/llama3_2_1B_cuda.yaml"

module load conda
conda activate ft-llms

srun --ntasks-per-node=2 \
     tune run \
     --nnodes 2 \
     --nproc_per_node 2 \
     --rdzv_id 1234 \
     --rdzv_backend c10d \
     --rdzv_endpoint "$head_node_ip:29500" \
     lora_finetune_distributed \
     --config "$CONFIG_PATH"