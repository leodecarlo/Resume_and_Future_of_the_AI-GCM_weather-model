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

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

path_mcrosta = r"/leonardo_work/DL4SF_Illumia_0/mcrosta/"

class NetCDFDataset(Dataset):
    def __init__(self, features_file_path, leadtime=1, start_year=1981, end_year=2019):
        self.features_file_path = features_file_path
        self.leadtime = leadtime
        self.start_year = start_year
        self.end_year = end_year
        self.variables = ['T_2M','U_10M','V_10M','PMSL','ASOB_S','ATHB_S']
        self.lenvariables = len(self.variables)
        self.min_value = np.load(f"{path_mcrosta}data_npy/norm_01_range/min_vec.npy")
        self.max_value = np.load(f"{path_mcrosta}data_npy/norm_01_range/max_vec.npy")

        # Store file paths instead of opening files
        self.file_paths = {}
        for year in range(self.start_year, self.end_year + 1):
            self.file_paths[year] = {}
            for variable in self.variables:
                features_file = os.path.join(self.features_file_path, f'{variable}_HLDRea_002_1hr_{year}.nc')
                self.file_paths[year][variable] = features_file

        self.index_map = self._create_index_map()

    def _create_index_map(self):
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
        return len(self.index_map)

    def normalize_minmax(self, value, indvar):
        return (value - self.min_value[indvar]) / (self.max_value[indvar] - self.min_value[indvar])

    def __getitem__(self, idx):
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

def main():
    
    # Read OpenMPI's rank and world size
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))

    # Set RANK and WORLD_SIZE for PyTorch
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    
    # Initialize the process group with 'nccl' backend
    dist.init_process_group(backend='nccl', init_method='env://')
   
    print(f"Hello from rank {rank} out of {world_size}")

    

     # Determine local rank for GPU assignment
    GPUS_PER_NODE = int(os.environ.get('GPUS_PER_NODE', 4))  # Default to 4 if not set
    local_rank = rank % GPUS_PER_NODE
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Print rank and device info for debugging (only for rank 0)
    if rank == 0:
        print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

    # Define directories
    features_path = '/leonardo_work/DL4SF_Illumia_0/data/raw/'

    # Create dataset and dataloader with DistributedSampler
    dataset = NetCDFDataset(features_path, leadtime=1, start_year=1981, end_year=2017)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, pin_memory=True, num_workers=4)

    val_dataset = NetCDFDataset(features_path, leadtime=1, start_year=2018, end_year=2019)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler, pin_memory=True, num_workers=4)

    # Define model, loss function, and optimizer
    model = Generator(input_channels=6, output_channels=6).to(device)
    initialize_weights(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss = []
    val_loss = []

    epochs = 4

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Ensure the model is in training mode
        sampler.set_epoch(epoch)  # Shuffle data each epoch

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} Training", ncols=100, position=rank)
        for ii, batch in progress_bar:
            inputs = batch['features'].to(device, non_blocking=True)  # Shape: [batch_size, 6, H, W]
            labels = batch['labels'].to(device, non_blocking=True)    # Shape: [batch_size, 6, H, W]

            optimizer.zero_grad()

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
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{epochs} Validation", ncols=100, position=rank)
            for jj, batch in progress_bar:
                inputs = batch['features'].to(device, non_blocking=True)  # Shape: [batch_size, 6, H, W]
                labels = batch['labels'].to(device, non_blocking=True)    # Shape: [batch_size, 6, H, W]

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

        np.save('/leonardo_work/DL4SF_Illumia2/UNET_Torch/prova_train_loss_mpi_nopref.npy', train_loss)
        np.save('/leonardo_work/DL4SF_Illumia2/UNET_Torch/prova_val_loss_mpi_nopref.npy', val_loss)

        # Visualization
        # Since num_workers=4, it's safe to iterate over the dataloader
        for batch in dataloader:
            input_images = batch['features'].data.float()
            target_images = batch['labels'].data.float()
            outputs = model.module(input_images.to(device))
            output_images = outputs.cpu().data.float()
            break

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