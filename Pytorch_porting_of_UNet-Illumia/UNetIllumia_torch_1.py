import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 


import sys
print(sys.path)


def pad_to(x, stride=2):
    """
    Pads the input tensor x so that its height and width are multiples of stride.

    Parameters:
    - x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
    - stride (int): The stride value to which height and width should be multiples.

    Returns:
    - padded (torch.Tensor): Padded tensor.
    - pads (tuple): Tuple of padding applied (pad_left, pad_right, pad_top, pad_bottom).
    """
    _, _, h, w = x.shape

    # Calculate new height and width to be multiples of stride
    new_h = h + (stride - h % stride) if h % stride != 0 else h
    new_w = w + (stride - w % stride) if w % stride != 0 else w

    # Calculate the total padding needed
    pad_h = new_h - h
    pad_w = new_w - w

    # Distribute padding equally on both sides (left/right, top/bottom)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Use F.pad to pad the tensor with zeros (mode='constant', value=0).
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    pads = (pad_left, pad_right, pad_top, pad_bottom)
    return padded, pads



def unpad(x, skip):
    """
    Removes padding from the input tensor x to match the spatial dimensions of the skip tensor.

    Parameters:
    - x (torch.Tensor): Tensor to be unpadded, shape [batch, channels, height_x, width_x].
    - skip (torch.Tensor): Reference tensor with desired spatial dimensions, shape [batch, channels, height_skip, width_skip].

    Returns:
    - x (torch.Tensor): Unpadded tensor matching the height and width of skip.
    """
    _, _, h_x, w_x = x.shape
    _, _, h_skip, w_skip = skip.shape

    diff_h = h_x - h_skip
    diff_w = w_x - w_skip

    if diff_h > 0:
        pixels_top = diff_h // 2
        pixels_bottom = diff_h - pixels_top
        if pixels_bottom == 0:
            x = x[:, :, pixels_top:, :]
        else:
            x = x[:, :, pixels_top:-pixels_bottom, :]
    if diff_w > 0:
        pixels_left = diff_w // 2
        pixels_right = diff_w - pixels_left
        if pixels_right == 0:
            x = x[:, :, :, pixels_left:]
        else:
            x = x[:, :, :, pixels_left:-pixels_right]
    return x


"""
TO DO:  DEFINE OUR NORMALIZATION AFTER EACH LAYER: i.e. NORMALIZE THE OUTPUT  AT THE END OF A LAYER BEFORE THE NEXT ONE(THAT IS AFTER ACTIVATION FUNC.)
"""

class Downsample(nn.Module):
    def __init__(self, input_channels, filters, size=4, apply_batchnorm=False):
        """
        Parameters:
        - input_channels (int): Number of input channels.
        - filters (int): Number of output filters.
        - size (int): Kernel size for the convolutional layer.
        - apply_batchnorm (bool): Whether to include batch normalization.
        """
        
        super(Downsample, self).__init__()
        
        layers = [nn.Conv2d(in_channels=input_channels,out_channels=filters, kernel_size=size, stride=2, padding=1, bias=False),]
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        
        # Activation Function
        layers.append(nn.LeakyReLU(0.2))
        
        # Stack layers sequentially
        self.model = nn.Sequential(*layers)
        
        # Initialize weights after defining self.model
        #nn.init.normal_(self.model[0].weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        """
        Forward pass through the downsampling block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
        - torch.Tensor: Output tensor after applying the downsampling block.
        """
        return self.model(x)


class Upsample(nn.Module):
    def __init__(self, input_channels, filters, size=4, apply_batchnorm=False, apply_dropout=False):
        """
        Parameters:
        - input_channels (int): Number of input channels.
        - filters (int): Number of output filters.
        - size (int): Kernel size for the transposed convolutional layer.
        - apply_batchnorm (bool): Whether to include batch normalization.
        - apply_dropout (bool): Whether to include dropout.
        """
        super(Upsample, self).__init__()
        
        
        # Define layers
        layers = [nn.ConvTranspose2d(in_channels=input_channels, out_channels=filters, kernel_size=size, stride=2, padding=1, bias=False),]
        
        # Optional Batch Normalization
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        
        # Optional Dropout
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        
        # Activation Function
        layers.append(nn.LeakyReLU(0.2))
        
        # Stack layers sequentially
        self.model = nn.Sequential(*layers)
        
        # Initialize weights after defining self.model
        #nn.init.normal_(self.model[0].weight, mean=0.0, std=0.02)
        
              
    def forward(self, x):
        """
        Forward pass through the upsampling block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
        - torch.Tensor: Output tensor after applying the upsampling block.
        """
        
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        """
        Constructs the Generator model with downsampling and upsampling paths.

        Parameters:
        - output_channels (int): Number of output channels.
        """
        super(Generator,self).__init__()
        
        self.down_stack = nn.ModuleList([
            Downsample(input_channels=input_channels, filters=64, size=4, apply_batchnorm=False),  # (340, 268, 64)
            Downsample(input_channels=64, filters=128, size=4),  # (170, 134, 128)
            Downsample(input_channels=128, filters=256, size=4),  # (85, 67, 256)
            Downsample(input_channels=256, filters=512, size=4),  # (43, 34, 512)
            Downsample(input_channels=512, filters=512, size=4),  # (22, 17, 512)
            Downsample(input_channels=512, filters=512, size=4),  # (11, 9, 512)
        ])
        
        self.up_stack = nn.ModuleList([
            Upsample(input_channels=512, filters=512, size=4),  # (22, 17, 1024)
            Upsample(input_channels=1024, filters=512, size=4),  # (43, 34, 1024)
            Upsample(input_channels=1024, filters=256, size=4),  # (85, 67, 512)
            Upsample(input_channels=512, filters=128, size=4),  # (170, 134, 256)
            Upsample(input_channels=256, filters=64, size=4),   # (340, 268, 128)
        ])
        
        #Define final transposed convolutional layer
        self.last = nn.ConvTranspose2d(in_channels=128,out_channels=output_channels,kernel_size=4,stride=2,padding=1,bias=False)        

        #Initialize weights for the final layer
        #nn.init.normal_(self.last.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        
        original_shape = x  # Store original input shape

        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            x, pads = pad_to(x, stride=2)  # Ensures dimensions are multiples of stride
            skips.append(x)
            
        #Reverse the skips list and exclude the last one
        skips = list(reversed(skips[:-1]))
        

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = unpad(x, skip)  # Removes padding to match skip connection dimensions
            x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
        
        # Final layer
        x = self.last(x)
        #x = torch.tanh(x) #for a different final layer activation

        x = unpad(x, original_shape)  # Removes any residual padding to match original input dimensions
        
        return x
        

def initialize_weights(model):
    """
    Initializes the weights of the model with a normal distribution.

    Parameters:
    - model (nn.Module): The PyTorch model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.constant_(m.bias, 0)


"""
# Creation of the model and testing with simulated input data
OUTPUT_CHANNELS = 1  # Desired number of output channels (e.g., segmentation classes)
model = Generator(output_channels=OUTPUT_CHANNELS)  # Initialize the Generator model with 6 output channels
initialize_weights(model)  # Apply weight initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
model.to(device)  # Transfer the model to the selected device (CPU/GPU)

# Simulate an input tensor with batch size 4, 6 channels, and dimensions 680x535
input_data = torch.randn(4, OUTPUT_CHANNELS, 680, 535).to(device)  # Simulated input
print(f"Input Shape: {input_data.shape}")  # Check the shape of the input

# Pass the input through the model
output_data = model(input_data)  # Model output

print(f"Output Shape: {output_data.shape}")  # Check the shape of the output, should be similar to the input

# Visualization of the input and output
input_image = input_data[0, 0, :, :].cpu().detach().numpy()  # Select the first channel of the first image in the batch
output_image = output_data[0, 0, :, :].cpu().detach().numpy()  # Select the first channel of the first output image

# Plot the input and output
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image (Channel 0)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Output Image (Channel 0)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_image - input_image, cmap='gray')
plt.title('Difference Image (Output - Input)')
plt.axis('off')

plt.savefig('/leonardo_work/DL4SF_Illumia_0/UNET_Torch/testUnetTorch.png')
print("Test plot saved as testUnetTorch.png")
plt.show()

"""