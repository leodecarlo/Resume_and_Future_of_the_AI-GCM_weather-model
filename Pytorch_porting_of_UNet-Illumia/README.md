# PyTorch Porting of UNet-Illumia

This directory contains the PyTorch implementation of the UNet-Illumia Neural Network [UNetIllumia_torch_1.py](UNetIllumia_torch_1.py). There are also different training files  with different HPC communications and their respective Slurm files (to run the scripts on the cluster). The training implements data-parallelism to move the data between the data-storage and the computing nodes, in this way  minimal resources of memory of the computing nodes are used to store data because just the single sample for the current computation is moved to the nodes instead of preloading the entire dataset (the files that implement data-prefetching move two samples: [prova_torch_v4.py](prova_torch_v4.py) and [prova_torch_v4_mpi.py](prova_torch_v4_mpi.py) ).  This allows to train the model with Large Datasets without troubles of memory.


Below there is a description of the differences between different files.
Basically we will use the `mpi` versions, because they don't give problems with multinode runs ( while the ones based on `Nvidia-nccl` backend   [prova_torch_v3.py](prova_torch_v3.py) and [prova_torch_v4.py](prova_torch_v4.py) give troubles ).

---


## 1. Python Scripts

1. [UNetIllumia_torch_1.py](UNetIllumia_torch_1.py) is the actual architeture "UNet-like generator" that we used in the first part of the project. Implements skip connections and encoder-decoder structure.

2.  **`prova_torch_v3.py`** :
- Description:  implements the single-node PyTorch training pipeline for the UNet-Illumia model `UNetIllumia_torch_1.py` .
- Key features:
  - Store file paths instead of laoding data in the computing nodes, map dataset indices to specific time steps and years and retrieve a single data sample from the dataset based on the provided index idx;
  - The file is set to run on multi-GPU and multinode with the Ǹvidia-nccl`backend (theoretically, but this gives troubles with multunode).

3. **`prova_torch_v4.py`** :
- Description: this does the same of `prova_torch_v3.py` with the addition of  implementing Asynchronous Data Prefetching to GPU. Asynchronous Data Prefetching involves loading the next batch of data and  transferring it to the GPU while the current batch is being processed by the model. This can reduce idle time for the GPU and improve training speed.
- Key features: add batch prefetching.

4.  **`prova_torch_v3_mpi.py`** :
- Description: this does the same of `prova_torch_v3.py` but the environment to run on multiGPU and multinode is set with `mpi` .
- Key features: the mpi environment scale well on multinode (test up to 8 nodes).

 
5. **`prova_torch_v4_mpi.py`** :
- Description: This does the same of `prova_torch_v4.py` but the environment to go on multiGPU and multinode is set with `mpi` .
- Key features: batch prefetching and the mpi environment scale well on multinode (test up to 8 nodes).

---

## 2. Slurm Scripts

1.**`GPUslurm_test.sh`**
- Description: A Slurm script to run the model on a single-GPU-node with torchrun.

2. `GPUslurm_test_2.sh`
- Description: A modified Slurm script for multi-GPU-node with torchrun (gave problems).

3. `GPUslurm_mpirun.sh`
- Description: A Slurm script for multi-GPU-node with `mpi`, worked well.


