#!/bin/bash
#SBATCH --account=cin_staff
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # Number of tasks per node (aligns with GPUs)
#SBATCH --gres=gpu:4                 # Request 4 GPUs per node
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg           # Uncomment if needed
#SBATCH --time=00:12:00
#SBATCH --exclusive

# Set variables
GPUS_PER_NODE=4
MASTER_PORT=29500                   # Ensure this port is free or change to another

echo "NODELIST=${SLURM_NODELIST}"

# Determine the master address (the first node)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

# Total number of tasks across all nodes
#export WORLD_SIZE=$SLURM_NTASKS
#echo "WORLD_SIZE=$WORLD_SIZE"

# Each task is assigned a rank based on SLURM_PROCID
#export RANK=$SLURM_PROCID
echo "RANK=$RANK"

# Load necessary modules
module purge
module load gcc
module load openmpi
module load cuda
module load cudnn
module load nccl

# Activate your virtual environment
source /leonardo_work/DL4SF_Illumia_0/UNET_Torch/UNET_env/bin/activate
#source /leonardo_work/DL4SF_Illumia_0/ldecarlo/OpenSTL/openstl_env/bin/activate


# Set the number of OpenMP threads to the number of CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo " SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK "

echo " SLURM_NTASKS=$SLURM_NTASKS "



# Launch the distributed training using mpirun
mpirun -n $SLURM_NTASKS \
       --bind-to none \
       --map-by slot \
       -x MASTER_ADDR=$MASTER_ADDR \
       -x MASTER_PORT=$MASTER_PORT \
       -x OMP_NUM_THREADS \
       $(which python) /leonardo_work/DL4SF_Illumia_0/UNET_Torch/prova_torch_v4_mpi.py
       
       
       #$(which python) /leonardo_work/DL4SF_Illumia_0/UNET_Torch/hello_ddp.py