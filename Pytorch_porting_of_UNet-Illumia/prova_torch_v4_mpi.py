
#Asynchronous data prefetching involves loading the next batch of data and 
# transferring it to the GPU while the current batch is being processed by the model. 
# This can reduce idle time for the GPU and improve training speed.


import os
import torch
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import torch.nn as nn
from UNetIllumia_torch_1 import Generator
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler #For splitting the dataset across multiple processes.
import torch.distributed as dist #For initializing and managing the distributed environment.

path_mcrosta = r"/leonardo_work/DL4SF_Illumia_0/mcrosta/"

#implement asynchronous data prefetching to the GPU: It wraps around a standard PyTorch DataLoader and manages the prefetching of data batches,
# transferring them to the GPU in the background while the model is processing the current batch.

class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device) #A new CUDA stream created for asynchronous data transfers.

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.prefetch()
        return self

    def __next__(self):
        # Wait for the previous data loading to finish
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        # Move data to current stream
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k].record_stream(torch.cuda.current_stream(self.device))
        # Prefetch next batch
        self.prefetch()
        return batch

    def prefetch(self):
        try:
            # Load next batch
            next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        # Transfer to GPU in the background
        with torch.cuda.stream(self.stream):
            for k in next_batch:
                if torch.is_tensor(next_batch[k]):
                    next_batch[k] = next_batch[k].to(self.device, non_blocking=True)
        self.next_batch = next_batch



class NetCDFDataset(Dataset):
    def __init__(self, features_file_path, leadtime=1, start_year=1980, end_year=2020):
        self.features_file_path = features_file_path
        self.leadtime = leadtime  #The forecast lead time (default is 1 hour).
        self.start_year = start_year
        self.end_year = end_year
        self.variables = ['T_2M','U_10M','V_10M','PMSL','ASOB_S','ATHB_S']
        self.lenvariables = len(self.variables) # Number of variables.
        self.min_value = np.load(f"{path_mcrosta}data_npy/norm_01_range/min_vec.npy")
        self.max_value = np.load(f"{path_mcrosta}data_npy/norm_01_range/max_vec.npy")

        # Store file paths instead of opening files
        self.file_paths = {} #Initializes a dictionary to store file paths for each variable and year
        for year in range(self.start_year, self.end_year + 1):
            self.file_paths[year] = {} # For each year, we create another dictionary to hold file paths for each variable.
            for variable in self.variables:
                features_file = os.path.join(self.features_file_path, f'{variable}_HLDRea_002_1hr_{year}.nc') #: Constructs the full file path by joining the base path with the filename
                self.file_paths[year][variable] = features_file

        self.index_map = self._create_index_map()

    def _create_index_map(self):    #method to map dataset indices to specific time steps and years.
        index_map = []
        for year in range(self.start_year, self.end_year + 1):
            sample_variable = self.variables[0]
            features_file = self.file_paths[year][sample_variable]
            with nc.Dataset(features_file, 'r') as ds:
                num_time_steps = ds.variables[sample_variable].shape[0]
            num_samples = num_time_steps - self.leadtime
            for time_idx in range(num_samples):
                index_map.append((year, time_idx))
        return index_map

    def __len__(self):
        print(len(self.index_map))
        return len(self.index_map)

    def normalize_minmax(self, value, indvar):
        return (value - self.min_value[indvar]) / (self.max_value[indvar] - self.min_value[indvar])

    def __getitem__(self, idx): # Retrieve a single data sample from the dataset based on the provided index idx
        year, time_idx = self.index_map[idx]
        features = []
        labels = []
        for indvar, variable in enumerate(self.variables):
            features_file = self.file_paths[year][variable]
            with nc.Dataset(features_file, 'r') as ds:
                var_data = ds.variables[variable]
                feature = var_data[time_idx, :, :]
                label = var_data[time_idx + self.leadtime, :, :]
                # Normalize
                feature = self.normalize_minmax(feature, indvar)
                label = self.normalize_minmax(label, indvar)
                features.append(feature)
                labels.append(label)
        # Stack features and labels along the channel dimension
        features = np.stack(features, axis=0)  # Shape: [6, H, W]
        labels = np.stack(labels, axis=0)      # Shape: [6, H, W]
        features = torch.tensor(features).float()
        labels = torch.tensor(labels).float()
        return {'features': features, 'labels': labels}

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.constant_(m.bias, 0)
            
def worker_init_fn(worker_id):
    #The torch.initial_seed() can produce a 64-bit integer, but np.random.seed() and random.seed() require 32-bit integers.
    #Taking modulo 2**32 ensures the seed is within the acceptable range.
    seed = torch.initial_seed() % 2**32 
    np.random.seed(seed)
    random.seed(seed)            


def main():
    # Initialize the process group with 'nccl' backend (optimized for NVIDIA GPUs)
    dist.init_process_group(backend='nccl')

    # Get rank and world size from torchrun's environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get local rank from torchrun's environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device based on local_rank
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Print rank and device info for debugging (only for rank 0)
    if rank == 0:
        print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

    # Define directories
    features_path = '/leonardo_work/DL4SF_Illumia_0/data/raw/'

    # Create dataset and dataloader with DistributedSampler
    dataset = NetCDFDataset(features_path, leadtime=1, start_year=1981, end_year=2017)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) #ensures each process in the distributed training setup receives a unique subset of the data
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, pin_memory=True, num_workers=4,persistent_workers=True, worker_init_fn=worker_init_fn)

    val_dataset = NetCDFDataset(features_path, leadtime=1, start_year=2018, end_year=2019)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler, pin_memory=True, num_workers=4, persistent_workers=True, worker_init_fn=worker_init_fn)

    # Define model, loss function, and optimizer
    model = Generator(input_channels=6, output_channels=6).to(device)
    initialize_weights(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss = []
    val_loss = []

    prefetch_train_loader = PrefetchLoader(dataloader, device)
    prefetch_val_loader = PrefetchLoader(val_dataloader, device)
    
    epochs = 4
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Ensure the model is in training mode
        sampler.set_epoch(epoch)  # Shuffle data each epoch

        '''
        Visualizing the Flow

            Dataset:
                Returns individual samples as dictionaries: {'features': ..., 'labels': ...}.

            DataLoader:
                Batches these samples into larger dictionaries: {'features': [batch_size, 6, H, W], 'labels': [batch_size, 6, H, W]}.

            enumerate(dataloader):
                Wraps each batch with an index: (0, batch_0), (1, batch_1), etc.

            tqdm:
                Displays the progress of these (index, batch) iterations.
        '''

        progress_bar = tqdm(enumerate(prefetch_train_loader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} Training", ncols=100, position=rank)        
        for ii, batch in progress_bar:
            inputs = batch['features'] # Shape: [batch_size, 6, H, W]
            labels = batch['labels']   # Shape: [batch_size, 6, H, W]

            if ii == 0:
                print(f"batch index:{ii}")
                print(f"features shape: {batch['features'].shape}")
                print(f"labels shape: {batch['labels'].shape}")
            
            optimizer.zero_grad() # In PyTorch, gradients accumulate by default. optimizer.zero_grad() resets the gradients of all model parameters.Ensures that gradients from the previous batch do not affect the current update

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate average loss
        epoch_loss = running_loss / len(dataloader.dataset)
        train_loss.append(epoch_loss)

        # Validation loop
        val_running_loss = 0.0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            progress_bar = tqdm(enumerate(prefetch_val_loader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{epochs} Validation", ncols=100, position=rank)            
            for _, batch in progress_bar:
                inputs = batch['features']  # Shape: [batch_size, 6, H, W]
                labels = batch['labels']    # Shape: [batch_size, 6, H, W]

                outputs = model(inputs)
                val_loss_batch = criterion(outputs, labels)
                val_running_loss += val_loss_batch.item() * inputs.size(0)

                progress_bar.set_postfix({'val_loss': val_loss_batch.item()})

        # Calculate average validation loss
        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        val_loss.append(val_epoch_loss)

        if rank == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_epoch_loss:.4f}')

    # Save loss arrays and plots (only on rank 0 to avoid conflicts)
    if rank == 0:
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)

        np.save('/leonardo_work/DL4SF_Illumia_0/UNET_Torch/prova_train_loss_torchrun_pref.npy', train_loss)
        np.save('/leonardo_work/DL4SF_Illumia_0/UNET_Torch/prova_val_loss_torchrun_pref.npy', val_loss)

        # Visualization
        # Since num_workers=4, it's safe to iterate over the prefetch_train_loader (which gives back batches)
        # and detach to avoid gradients
        model.eval()
        with torch.no_grad():
            for batch in prefetch_train_loader:
                input_images = batch['features'].detach().cpu().float()
                target_images = batch['labels'].detach().cpu().float()
                outputs = model.module(batch['features']).detach().cpu().float()
                output_images = outputs
                break  # Only visualize the first batch

        # Convert to NumPy arrays
        input_image_np = input_images[0, 0, :, :].cpu().numpy()
        target_image_np = target_images[0, 0, :, :].cpu().numpy()
        output_image_np = output_images[0, 0, :, :].cpu().numpy()

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        input_plot = axes[0].imshow(input_image_np, cmap='viridis',origin = 'lower')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        fig.colorbar(input_plot, ax=axes[0])

        target_plot = axes[1].imshow(target_image_np, cmap='viridis', origin = 'lower')
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        fig.colorbar(target_plot, ax=axes[1])

        output_plot = axes[2].imshow(output_image_np, cmap='viridis', origin = 'lower')
        axes[2].set_title('Output Image')
        axes[2].axis('off')
        fig.colorbar(output_plot, ax=axes[2])

        plt.savefig('/leonardo_work/DL4SF_Illumia2/UNET_Torch/prova_plot_mpi_nopref.png')
        plt.close(fig)  # Close the figure to free memory

    # Shutdown the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()