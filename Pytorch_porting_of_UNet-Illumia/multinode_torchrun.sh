#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -N 6
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH -A cin_staff
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

###SBATCH --qos=boost_qos_dbg


export TMPDIR="$CINECA_SCRATCH/tmp" #### < change the folder here once
export TEMPDIR="$CINECA_SCRATCH/tmp" #### < change the folder here once
export TMP="$CINECA_SCRATCH/tmp" #### < change the folder here once
export TEMP="$CINECA_SCRATCH/tmp" #### < change the folder here once
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib
 
 
echo $TMPDIR
 
#module load profile/deeplrn
#module load cineca-ai

# Load necessary modules
module purge
module load cuda cudnn nccl openmpi

# Activate your virtual environment
source /leonardo_work/DL4SF_Illumia_0/UNET_Torch/UNET_env/bin/activate

 
echo 'all modules loaded'

echo 'I am multinode_torchrun.sh'

 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
 
echo Node IP: $head_node_ip
export LOGLEVEL=INFO
 
srun torchrun --nproc_per_node=4 --nnodes=6 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
prova_torch_v3.py