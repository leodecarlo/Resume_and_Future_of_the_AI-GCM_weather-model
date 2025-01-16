#!/bin/bash
#SBATCH --account=cin_staff
#SBATCH --nodes=1
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg  # For debugging
#SBATCH --time=00:30:00
#SBATCH --exclusive

GPUS_PER_NODE=4

echo "NODELIST=${SLURM_NODELIST}"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_NTASKS=$SLURM_NTASKS"
NTASKS_PER_NODE=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
echo "NTASKS_PER_NODE=$NTASKS_PER_NODE"
export WORLD_SIZE=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "WORLD_SIZE=$WORLD_SIZE"
MASTER_PORT=29500  # Ensure this port is free or change to another


# Load necessary modules
module purge
module load cuda cudnn nccl openmpi

# Activate your virtual environment
source /leonardo_work/DL4SF_Illumia_0/UNET_Torch/UNET_env/bin/activate

# Set the number of OpenMP threads to the number of CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Export necessary environment variables
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$(($SLURM_PROCID))

# Use torchrun for launching the distributed job
torchrun --nnodes=$SLURM_JOB_NUM_NODES \
         --nproc_per_node=$SLURM_NTASKS_PER_NODE \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         /leonardo_work/DL4SF_Illumia_0/UNET_Torch/prova_torch_v3.py