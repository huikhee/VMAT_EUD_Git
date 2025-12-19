# Combined u-net and decoder architecture
# cross training
# same as used for furst paper submission
#now adapted for VMAT
import sys
import math

import numpy as np
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F 


from torch.utils.data import DataLoader,Dataset,random_split,Subset
#from torchvision import transforms
#from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from torch.optim import AdamW

import scipy.io
from scipy.signal import convolve2d
from scipy.io import loadmat

#from torch.utils.tensorboard import SummaryWriter


import random
from random import randint

import platform
import sys
import pandas as pd
#import sklearn as sk

import matplotlib.pyplot as plt

import os
import socket

from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()

device = torch.device("mps") if torch.backends.mps.is_built() \
    else torch.device("cuda") if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
#print(f"Scikit-Learn {sk.__version__}")
print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

########################################################################################

# Generate data and save

def generate_random_vectors_scalar_regular(seed):
    """Generate random vectors and scalars with regular pattern."""
    np.random.seed(seed)
    
    num_samples = 2048
    vector_length = 52

    # Initialize arrays
    scalar1 = np.zeros(num_samples)
    scalar2 = np.zeros(num_samples)
    scalar3 = np.zeros(num_samples)
    vector1 = np.ones((num_samples, vector_length)) * -20
    vector2 = np.ones((num_samples, vector_length)) * 20
    vector1_weight = np.ones((num_samples, vector_length)) * 0.5
    vector2_weight = np.ones((num_samples, vector_length)) * 0.5

    for i in range(num_samples):
        if i == 0:
            prev_scalar2 = np.random.uniform(-130, 110.05)
            prev_scalar3 = np.around(np.random.uniform(prev_scalar2 + 20, 130.05), 1)
            prev_vector1_s = np.random.uniform(-130, 110.05)
            prev_vector2_s = np.around(np.random.uniform(prev_vector1_s + 20, 130.05), 1)
        else:
            prev_scalar2 = scalar2[i - 1]
            prev_scalar3 = scalar3[i - 1]
            prev_vector1_s = vector1_s
            prev_vector2_s = vector2_s

        scalar1[i] = np.around(np.random.uniform(1, 40), 1)
        scalar2[i] = np.around(np.random.uniform(max(prev_scalar2 - 20, -130), min(prev_scalar2 + 20, 110.05)), 1)
        min_value = scalar2[i] + 20
        scalar3[i] = np.around(np.random.uniform(max(min_value, prev_scalar3 - 20), min(prev_scalar3 + 20, 130.05)), 1)

        lower_limit = int(np.ceil((130 + scalar2[i]) / 5))
        upper_limit = int(np.ceil((130 + scalar3[i]) / 5))

        lower_limit_weight = max(0, lower_limit - 2)
        upper_limit_weight = min(52, upper_limit + 2)

        vector1_weight[i, lower_limit_weight:upper_limit_weight] = 1
        vector2_weight[i, lower_limit_weight:upper_limit_weight] = 1

        vector1_s = np.around(np.random.uniform(max(prev_vector1_s - 40, -130), min(prev_vector1_s + 40, 110.05)), 1)
        vector2_s = np.around(np.random.uniform(max(vector1_s + 20, prev_vector2_s - 40), min(prev_vector2_s + 40, 130.05)), 1)

        for j in range(lower_limit, upper_limit):
            vector1[i, j] = vector1_s
            vector2[i, j] = vector2_s

        # Handle boundary conditions
        if lower_limit - 1 > 0:
            vector1[i, lower_limit - 1] = vector1[i, lower_limit]
            vector2[i, lower_limit - 1] = vector2[i, lower_limit]
        if lower_limit - 2 > 0:
            vector1[i, lower_limit - 2] = vector1[i, lower_limit]
            vector2[i, lower_limit - 2] = vector2[i, lower_limit]
        if upper_limit < 52:
            vector1[i, upper_limit] = vector1[i, upper_limit - 1]
            vector2[i, upper_limit] = vector2[i, upper_limit - 1]
        if upper_limit + 1 < 52:
            vector1[i, upper_limit + 1] = vector1[i, upper_limit - 1]
            vector2[i, upper_limit + 1] = vector2[i, upper_limit - 1]

    return vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight

def generate_random_vectors_scalar_semiregular(seed):
    """Generate random vectors and scalars with semi-regular pattern."""
    np.random.seed(seed)
    
    num_samples = 2048
    vector_length = 52

    scalar1 = np.zeros(num_samples)
    scalar2 = np.zeros(num_samples)
    scalar3 = np.zeros(num_samples)
    vector1 = np.ones((num_samples, vector_length)) * -20
    vector2 = np.ones((num_samples, vector_length)) * 20
    vector1_weight = np.ones((num_samples, vector_length)) * 0.5
    vector2_weight = np.ones((num_samples, vector_length)) * 0.5

    for i in range(num_samples):
        if i == 0:
            prev_scalar2 = np.random.uniform(-130, 120.05)
            prev_scalar3 = np.around(np.random.uniform(prev_scalar2 + 10, 130.05), 1)
            prev_vector1_s = np.random.uniform(-130, 120.05)
            prev_vector2_s = np.around(np.random.uniform(prev_vector1_s + 10, 130.05), 1)
        else:
            prev_scalar2 = scalar2[i - 1]
            prev_scalar3 = scalar3[i - 1]
            prev_vector1_s = vector1[i-1, lower_limit]
            prev_vector2_s = vector2[i-1, lower_limit]

        scalar1[i] = np.around(np.random.uniform(1, 40), 1)
        scalar2[i] = np.around(np.random.uniform(max(prev_scalar2 - 20, -130), min(prev_scalar2 + 20, 120.05)), 1)
        min_value = scalar2[i] + 10
        scalar3[i] = np.around(np.random.uniform(max(min_value, prev_scalar3 - 20), min(prev_scalar3 + 20, 130.05)), 1)

        lower_limit = int(np.ceil((130 + scalar2[i]) / 5))
        upper_limit = int(np.ceil((130 + scalar3[i]) / 5))

        lower_limit_weight = max(0, lower_limit-2)
        upper_limit_weight = min(52, upper_limit+2)

        vector1_weight[i, lower_limit_weight:upper_limit_weight] = 1.0
        vector2_weight[i, lower_limit_weight:upper_limit_weight] = 1.0

        for j in range(lower_limit, upper_limit):
            if j == lower_limit:
                vector1[i, j] = np.around(np.random.uniform(max(prev_vector1_s - 40, -130), min(prev_vector1_s + 40, 120.05)), 1)
                vector2[i, j] = np.around(np.random.uniform(max(vector1[i, j] + 10, prev_vector2_s - 40), min(prev_vector2_s + 40, 130.05)), 1)
            else:
                min_value = max(vector1[i, j-1] - 10, -130)
                max_value = min(vector1[i, j-1] + 10, 120.05)
                vector1[i, j] = np.around(np.random.uniform(min_value, max_value), 1)
                
                min_value = max(vector1[i, j] + 10, -130)
                max_value1 = min(vector2[i, j-1] - 10, 130.05)
                max_value2 = min(vector2[i, j-1] + 10, 130.05)
                max_value = np.around(np.random.uniform(max_value1, max_value2), 1)
                vector2[i, j] = max(min_value, max_value)

                vector1[i, j] = np.around(np.random.uniform(max(vector1[i-1, j] - 40, vector1[i, j]), min(vector1[i-1, j] + 40, vector1[i, j])), 1)
                vector2[i, j] = np.around(np.random.uniform(max(vector2[i-1, j] - 40, vector2[i, j]), min(vector2[i-1, j] + 40, vector2[i, j])), 1)

        # Handle boundary conditions
        if lower_limit - 1 > 0:
            vector1[i, lower_limit-1] = vector1[i, lower_limit]
            vector2[i, lower_limit-1] = vector2[i, lower_limit]
        if lower_limit - 2 > 0:
            vector1[i, lower_limit-2] = vector1[i, lower_limit]
            vector2[i, lower_limit-2] = vector2[i, lower_limit]
        if upper_limit < 52:
            vector1[i, upper_limit] = vector1[i, upper_limit-1]
            vector2[i, upper_limit] = vector2[i, upper_limit-1]
        if upper_limit + 1 < 52:
            vector1[i, upper_limit+1] = vector1[i, upper_limit-1]
            vector2[i, upper_limit+1] = vector2[i, upper_limit-1]

    return vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight


def generate_random_vectors_scalars(seed):
    """Generate random vectors and scalars for VMAT data."""
    np.random.seed(seed)
    
    num_samples = 2048
    vector_length = 52

    # Initialize arrays
    scalar1 = np.zeros(num_samples)
    scalar2 = np.zeros(num_samples)
    scalar3 = np.zeros(num_samples)
    vector1 = np.ones((num_samples, vector_length)) * -20
    vector2 = np.ones((num_samples, vector_length)) * 20
    vector1_weight = np.ones((num_samples, vector_length)) * 0.5
    vector2_weight = np.ones((num_samples, vector_length)) * 0.5

    for i in range(num_samples):
        # Initialize previous values
        if i == 0:
            prev_scalar2 = np.random.uniform(-130, 120.05)
            prev_scalar3 = np.around(np.random.uniform(prev_scalar2 + 10, 130.05), 1)
            prev_vector1_s = np.random.uniform(-130, 120.05)
            prev_vector2_s = np.around(np.random.uniform(prev_vector1_s + 10, 130.05), 1)
        else:
            prev_scalar2 = scalar2[i - 1]
            prev_scalar3 = scalar3[i - 1]
            prev_vector1_s = vector1[i-1, lower_limit]
            prev_vector2_s = vector2[i-1, lower_limit]

        # Generate scalar values
        scalar1[i] = np.around(np.random.uniform(1, 40), 1)
        scalar2[i] = np.around(np.random.uniform(max(prev_scalar2 - 20, -130), 
                                               min(prev_scalar2 + 20, 120.05)), 1)
        
        min_value = scalar2[i] + 10
        scalar3[i] = np.around(np.random.uniform(max(min_value, prev_scalar3 - 20), 
                                               min(prev_scalar3 + 20, 130.05)), 1)

        # Calculate limits
        lower_limit = int(np.ceil((130 + scalar2[i]) / 5))
        upper_limit = int(np.ceil((130 + scalar3[i]) / 5))
        
        # Set weights
        lower_limit_weight = max(0, lower_limit-2)
        upper_limit_weight = min(52, upper_limit+2)
        vector1_weight[i, lower_limit_weight:upper_limit_weight] = 1.0
        vector2_weight[i, lower_limit_weight:upper_limit_weight] = 1.0

        # Generate vector values
        for j in range(lower_limit, upper_limit):
            if j == lower_limit:
                vector1[i, j] = np.around(np.random.uniform(
                    max(prev_vector1_s - 40, -130),
                    min(prev_vector1_s + 40, 120.05)), 1)
                vector2[i, j] = np.around(np.random.uniform(
                    max(vector1[i, j] + 10, prev_vector2_s - 40),
                    min(prev_vector2_s + 40, 130.05)), 1)
            else:
                # Generate subsequent vector values with constraints
                min_value = max(vector1[i, j-1] - 50, -130)
                max_value = min(vector1[i, j-1] + 50, 120.05)
                vector1[i, j] = np.around(np.random.uniform(min_value, max_value), 1)
                
                min_value = max(vector1[i, j] + 10, -130)
                max_value1 = min(vector2[i, j-1] - 50, 130.05)
                max_value2 = min(vector2[i, j-1] + 50, 130.05)
                max_value = np.around(np.random.uniform(max_value1, max_value2), 1)
                vector2[i, j] = max(min_value, max_value)

                # Add constraint for variation within 40 units from previous i-th element
                vector1[i, j] = np.around(np.random.uniform(
                    max(vector1[i-1, j] - 40, vector1[i, j]),
                    min(vector1[i-1, j] + 40, vector1[i, j])), 1)
                vector2[i, j] = np.around(np.random.uniform(
                    max(vector2[i-1, j] - 40, vector2[i, j]),
                    min(vector2[i-1, j] + 40, vector2[i, j])), 1)

        # Handle boundary conditions
        if lower_limit - 1 > 0:
            vector1[i, lower_limit-1] = vector1[i, lower_limit]
            vector2[i, lower_limit-1] = vector2[i, lower_limit]
        if lower_limit - 2 > 0:
            vector1[i, lower_limit-2] = vector1[i, lower_limit]
            vector2[i, lower_limit-2] = vector2[i, lower_limit]
        if upper_limit < 52:
            vector1[i, upper_limit] = vector1[i, upper_limit-1]
            vector2[i, upper_limit] = vector2[i, upper_limit-1]
        if upper_limit + 1 < 52:
            vector1[i, upper_limit+1] = vector1[i, upper_limit-1]
            vector2[i, upper_limit+1] = vector2[i, upper_limit-1]

    return vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight

def create_boundary_matrix(vector1, vector2, scalar1, scalar2, scalar3):
    """Create boundary matrix from vectors and scalars."""
    # Convert to integers
    vector1_int = np.round(vector1).astype(int)
    vector2_int = np.round(vector2).astype(int)
    scalar1_int = scalar1
    scalar2_int = np.round(scalar2).astype(int)
    scalar3_int = np.round(scalar3).astype(int)

    num_samples = len(scalar2_int)
    matrix_collection = []
    
    for i in range(num_samples):
        # Initialize matrix
        matrix = np.zeros((261, 261))

        # Fill matrix based on vectors
        for bin_index in range(52):
            y_start = max(-130 + bin_index * 5, -130)
            y_end = min(y_start + 5, 130)
            matrix[y_start+130:y_end+130, 
                  vector1_int[i,bin_index]+130:vector2_int[i,bin_index]+130] = 1

        # Apply scalar boundaries
        matrix[:max(-130, int(scalar2_int[i])) + 130, :] = 0
        matrix[min(130, int(scalar3_int[i])) + 130:, :] = 0
        
        # Rotate matrix
        matrix = np.flipud(matrix)
        rotated_matrix = scipy.ndimage.rotate(matrix, 0, reshape=False, mode='constant', cval=0.0)
        matrix_collection.append(rotated_matrix)


    return matrix_collection

def interpolate_vectors(v1_start, v1_end, v2_start, v2_end, s2_start, s2_end, 
                       s3_start, s3_end, num_interpolations=0):
    """Interpolate between vector pairs."""
    interpolated_v1 = []
    interpolated_v2 = []
    interpolated_s2 = []
    interpolated_s3 = []
    
    for i in range(1, num_interpolations + 1):
        t = i / (num_interpolations + 1)
        
        interp_v1 = (1 - t) * v1_start + t * v1_end
        interp_v2 = (1 - t) * v2_start + t * v2_end
        interp_s2 = (1 - t) * s2_start + t * s2_end
        interp_s3 = (1 - t) * s3_start + t * s3_end
        
        interpolated_v1.append(interp_v1)
        interpolated_v2.append(interp_v2)
        interpolated_s2.append(interp_s2)
        interpolated_s3.append(interp_s3)
    
    return interpolated_v1, interpolated_v2, interpolated_s2, interpolated_s3

class CustomDataset(Dataset):
    """Custom dataset for VMAT data."""
    def __init__(self, vector1, vector2, scalar1, scalar2, scalar3, 
                 vector1_weight, vector2_weight, arrays):
        self.vector1 = torch.from_numpy(vector1).float()
        self.vector2 = torch.from_numpy(vector2).float()
        self.scalar1 = torch.from_numpy(scalar1).float()
        self.scalar2 = torch.from_numpy(scalar2).float()
        self.scalar3 = torch.from_numpy(scalar3).float()
        self.vector1_weight = torch.from_numpy(vector1_weight).float()
        self.vector2_weight = torch.from_numpy(vector2_weight).float()
        self.arrays = torch.from_numpy(np.array(arrays)).float()

    def __len__(self):
        return len(self.vector1) - 2

    def __getitem__(self, idx):
        idx += 2
        prev_idx = idx - 1

        v1 = torch.cat([self.vector1[prev_idx].unsqueeze(0), 
                       self.vector1[idx].unsqueeze(0)], dim=1)
        v2 = torch.cat([self.vector2[prev_idx].unsqueeze(0), 
                       self.vector2[idx].unsqueeze(0)], dim=1)
        v1_weight = torch.cat([self.vector1_weight[prev_idx].unsqueeze(0), 
                             self.vector1_weight[idx].unsqueeze(0)], dim=1)
        v2_weight = torch.cat([self.vector2_weight[prev_idx].unsqueeze(0), 
                             self.vector2_weight[idx].unsqueeze(0)], dim=1)

        scalar1 = self.scalar1[idx].unsqueeze(0).unsqueeze(0)
        scalar2_current = self.scalar2[idx].unsqueeze(0).unsqueeze(0)
        scalar2_previous = self.scalar2[prev_idx].unsqueeze(0).unsqueeze(0)
        scalar3_current = self.scalar3[idx].unsqueeze(0).unsqueeze(0)
        scalar3_previous = self.scalar3[prev_idx].unsqueeze(0).unsqueeze(0)

        scalars = torch.cat([scalar1, scalar2_previous, scalar2_current, 
                           scalar3_previous, scalar3_current], dim=1)

        arrays = self.arrays[idx].unsqueeze(0)
        arrays_p = self.arrays[prev_idx].unsqueeze(0)

        return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p


def save_dataset(dataset, dataset_num):
    """Function to save dataset"""
    os.makedirs("VMAT_Art_data", exist_ok=True)
    filename = os.path.join("VMAT_Art_data", f"Art_dataset_coll0_{dataset_num}.pt")
    torch.save(dataset, filename)

def load_dataset(dataset_num):
    """Function to load dataset"""
    filename = os.path.join("VMAT_Art_data", f"Art_dataset_coll0_{dataset_num}.pt")
    if os.path.exists(filename):
        try:
            return torch.load(filename)
        except Exception as e:
            print(f"Error loading dataset {dataset_num}: {str(e)}")
            return None
    return None

def generate_and_save_dataset(dataset_num, KM):
    """Generate and save a complete dataset."""
    # Choose the appropriate vector generation function based on dataset number
    if 0 <= dataset_num <= 79:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalar_regular(42 + dataset_num)
    elif 80 <= dataset_num <= 159:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalar_semiregular(42 + dataset_num)
    elif 160 <= dataset_num <= 319:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalars(42 + dataset_num)
    elif 320 <= dataset_num <= 399:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalar_regular(42 + dataset_num)
    elif 400 <= dataset_num <= 479:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalar_semiregular(42 + dataset_num)
    else:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalars(43 + dataset_num)

    num_samples = len(vector1)
    num_interpolations = 5

    print(f"Random MLC data {dataset_num} created")

    combined_matrix_collection = []

    for i in range(0, num_samples - 1):
        interpolated_v1, interpolated_v2, interpolated_s2, interpolated_s3 = \
            interpolate_vectors(vector1[i], vector1[i + 1], vector2[i], vector2[i + 1],
                              scalar2[i], scalar2[i + 1], scalar3[i], scalar3[i + 1],
                              num_interpolations=num_interpolations)

        combined_v1 = [vector1[i]] + interpolated_v1 + [vector1[i + 1]]
        combined_v2 = [vector2[i]] + interpolated_v2 + [vector2[i + 1]]
        combined_s2 = [scalar2[i]] + interpolated_s2 + [scalar2[i + 1]]
        combined_s3 = [scalar3[i]] + interpolated_s3 + [scalar3[i + 1]]
        combined_s1 = np.repeat(scalar1[i], num_interpolations + 2)

        combined_matrix_collection.extend(
            create_boundary_matrix(combined_v1, combined_v2, combined_s1, 
                                 combined_s2, combined_s3))

    combined_matrix_collection_tensor = torch.stack(
        [torch.tensor(m) for m in combined_matrix_collection]).float().to(device)

    KM_tensor = torch.tensor(KM).float().unsqueeze(0).unsqueeze(0).to(device)
    arrays_gpu = F.conv2d(combined_matrix_collection_tensor.unsqueeze(1), 
                         KM_tensor, padding='same')
    print(f"Arrays {dataset_num} created")

    new_size = (131, 131)
    arrays_gpu_131 = F.interpolate(arrays_gpu, size=new_size, 
                                 mode='bilinear', align_corners=False)
    arrays = arrays_gpu_131.cpu()

    noise_std = 0.005
    for j in range(len(arrays)):
        noise = torch.randn(arrays[j].shape) * noise_std
        arrays[j] += noise

    final_arrays_list = []
    samples_per_CP = num_interpolations + 2

    for j in range(len(arrays) // samples_per_CP):
        final_array = sum(arrays[j * samples_per_CP + k] * 
                         (scalar1[j+1] / samples_per_CP) 
                         for k in range(samples_per_CP))
        final_arrays_list.append(final_array.numpy())

    final_arrays = np.array([arrays[0].numpy()] + final_arrays_list)

    Art_dataset = CustomDataset(vector1, vector2, scalar1, scalar2, scalar3,
                              vector1_weight, vector2_weight, final_arrays)

    save_dataset(Art_dataset, dataset_num)
    return Art_dataset



############################################################

# models.py

# embedded EncoderUnet model

# -------------------------------------------------------------------------
# ResidualConvBlock
# -------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """
    A 2D residual block with:
      - two conv layers (3x3)
      - GroupNorm(1, C) after each conv
      - ReLU activation
      - optional shortcut if in_channels != out_channels
    This matches your updated version that uses GroupNorm to emulate LayerNorm behavior in CNNs.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        
        # First convolution (3x3), groupnorm, ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution (3x3), groupnorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(1, out_channels)
        
        # Shortcut / identity mapping
        # If in/out channels differ, use a 1x1 conv; otherwise do nothing.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass:
          1) Apply conv1 + norm1 + ReLU
          2) Apply conv2 + norm2
          3) Add the original 'x' (optionally projected) to the result
          4) ReLU again
        """
        identity = self.shortcut(x)  # Might be 1x1 conv if channels differ

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.relu(out)
        return out


# -------------------------------------------------------------------------
# EncoderBlock
# -------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    A UNet encoder block:
      1) A ResidualConvBlock for feature extraction
      2) A MaxPool2d for spatial downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        # A residual conv block to learn features
        self.conv_block = ResidualConvBlock(in_channels, out_channels)
        # 2x2 pooling to halve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Returns:
          f:  feature map after the conv block
          p:  pooled feature map for the next encoding stage
        """
        f = self.conv_block(x)  # (B, out_channels, H, W)
        p = self.pool(f)        # (B, out_channels, H/2, W/2)
        return f, p


# -------------------------------------------------------------------------
# DecoderBlock
# -------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    A UNet decoder block:
      1) ConvTranspose2d for upsampling
      2) Concatenate with skip features
      3) ResidualConvBlock to combine them
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # 2x upsampling via transposed convolution
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # After concatenation, the channel dimension is out_channels + skip_channels (= out_channels).
        # But "skip_channels" = out_channels in this standard pattern, so total is (out_channels * 2).
        self.conv_block = ResidualConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_features):
        """
        Args:
          x: upsampled feature map
          skip_features: feature map from the encoder
        Returns:
          A combined feature map after upsampling, concatenation, and residual conv block.
        """
        x = self.conv_transpose(x)                   # (B, out_channels, 2H, 2W)
        x = torch.cat((x, skip_features), dim=1)     # (B, out_channels*2, 2H, 2W)
        x = self.conv_block(x)
        return x


# -------------------------------------------------------------------------
# EncoderUNet
# -------------------------------------------------------------------------
class EncoderUNet(nn.Module):
    """
    v4: Same UNet image synthesis as v1, but uses a token-based Transformer
    encoder with a dedicated CLS aggregation token.

    Tokenization (total tokens = 108):
      - 1 CLS token (learned; used for pooling)
      - 1 MU token (MU)
      - 2 jaw tokens (jaw1 prev/curr, jaw2 prev/curr)
      - 104 leaf tokens:
          * leaves 1..52 from vector1: [leaf_i_prev, leaf_i_curr]
          * leaves 53..104 from vector2: [leaf_i_prev, leaf_i_curr]
    The CLS token output is projected to a 1×64×64 latent image.
    """
    def __init__(self, vector_dim, scalar_count):
        super(EncoderUNet, self).__init__()
        
        # Activation for general use
        self.relu = nn.ReLU(inplace=True)
        
        # -----------------------------------------------------------------
        # 1) Parameter (v1/v2/scalars) -> Transformer -> latent image
        # -----------------------------------------------------------------
        self.vector_dim = int(vector_dim)
        self.scalar_count = int(scalar_count)
        if self.vector_dim % 2 != 0:
            raise ValueError(f"vector_dim must be even (prev+curr); got {self.vector_dim}")
        self.leaf_bins = self.vector_dim // 2  # 104 -> 52

        self.d_model = 128
        nhead = 4
        ff_dim = 256

        # Dedicated CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Token embeddings by type
        self.mu_token_embed = nn.Linear(1, self.d_model)
        self.jaw_token_embed = nn.Linear(2, self.d_model)
        self.leaf_token_embed = nn.Linear(2, self.d_model)

        # Positional embedding for 108 tokens:
        # [CLS] + [MU] + [jaw1] + [jaw2] + [104 leaf tokens]
        self.num_tokens = 1 + 1 + 2 + self.vector_dim
        self.param_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.param_transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.param_norm = nn.LayerNorm(self.d_model)

        # Project transformer global token -> 64x64 latent image
        self.latent_to_image = nn.Linear(self.d_model, 64 * 64)

        nn.init.normal_(self.param_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # -----------------------------------------------------------------
        # 4) UNet encoder
        # -----------------------------------------------------------------
        self.encoder1 = EncoderBlock(in_channels=1,  out_channels=32)   # 64x64 -> 32x32
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64)   # 32x32 -> 16x16
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128)  # 16x16 -> 8x8
        
        # Bottleneck
        self.bottleneck = ResidualConvBlock(in_channels=128, out_channels=256)  # 8x8 => 8x8
        
        # -----------------------------------------------------------------
        # 5) UNet decoder
        # -----------------------------------------------------------------
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128) # 8x8  -> 16x16
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)  # 16x16 -> 32x32
        self.decoder1 = DecoderBlock(in_channels=64,  out_channels=32)  # 32x32 -> 64x64
        
        # Upsample from 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                      kernel_size=3, stride=2, 
                                      padding=1, output_padding=1)
        
        # Final conv from 16->1 channel
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        
        # Resize 128x128 -> 131x131
        self.final_resize = nn.Upsample(size=(131, 131), mode='bilinear', align_corners=False)

    def forward(self, vector1, vector2, scalars):
        """
        Args:
          vector1: [B, 1, 104] (or similar) => Flatten to [B, 104]
          vector2: [B, 1, 104] => Flatten to [B, 104]
          scalars: [B, 1, scalar_count], e.g. [B,1,5] => Squeeze => [B,5]
        
        Returns:
          (B, 1, 131, 131) single-channel output image
        """
        # -----------------------------------------------------------------
        # Parameter transformer: (v1/v2/scalars) -> latent_image (B,1,64,64)
        # -----------------------------------------------------------------
        B = vector1.size(0)
        vector1 = vector1.view(B, -1)  # [B, 104]
        vector2 = vector2.view(B, -1)  # [B, 104]

        # Scalars: [B,1,5] or [B,1,5,1] -> [B,5]
        if scalars.dim() == 4 and scalars.size(-1) == 1:
            scalars = scalars.squeeze(-1)
        if scalars.dim() == 3 and scalars.size(1) == 1:
            scalars = scalars.squeeze(1)

        # MU token: (B,1) -> (B,1,d_model)
        mu = scalars[:, 0:1]
        mu_token = self.mu_token_embed(mu).unsqueeze(1)

        # Jaw tokens: jaw1=[1,2], jaw2=[3,4] => (B,2,2) -> (B,2,d_model)
        jaw1 = scalars[:, 1:3]
        jaw2 = scalars[:, 3:5]
        jaw_feats = torch.stack([jaw1, jaw2], dim=1)
        jaw_tokens = self.jaw_token_embed(jaw_feats)

        # Leaf tokens: 104 tokens, each is [prev, curr]
        lb = self.leaf_bins  # 52
        v1_prev = vector1[:, :lb]
        v1_curr = vector1[:, lb:]
        v2_prev = vector2[:, :lb]
        v2_curr = vector2[:, lb:]

        leaf1 = torch.stack([v1_prev, v1_curr], dim=-1)  # (B,52,2)
        leaf2 = torch.stack([v2_prev, v2_curr], dim=-1)  # (B,52,2)
        leaf_feats = torch.cat([leaf1, leaf2], dim=1)    # (B,104,2)
        leaf_tokens = self.leaf_token_embed(leaf_feats)  # (B,104,d_model)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,d_model)

        tokens = torch.cat([cls, mu_token, jaw_tokens, leaf_tokens], dim=1)  # (B,108,d_model)
        tokens = tokens + self.param_pos_embed[:, :tokens.size(1), :]
        tokens = self.param_transformer(tokens)
        pooled = self.param_norm(tokens[:, 0, :])

        latent_image = self.latent_to_image(pooled).view(B, 1, 64, 64)
        
        # -----------------------------------------------------------------
        # UNet encoder
        # -----------------------------------------------------------------
        f1, p1 = self.encoder1(latent_image)  # 64->32
        f2, p2 = self.encoder2(p1)            # 32->16
        f3, p3 = self.encoder3(p2)            # 16->8
        
        # Bottleneck
        btl = self.bottleneck(p3)            # 8->8
        
        # -----------------------------------------------------------------
        # UNet decoder
        # -----------------------------------------------------------------
        u3 = self.decoder3(btl, f3)          # 8->16
        u2 = self.decoder2(u3, f2)           # 16->32
        u1 = self.decoder1(u2, f1)           # 32->64
        
        # Upsample from 64->128
        up1 = self.up1(u1)  # => [B,16,128,128]
        up1 = self.relu(up1)
        
        # Final conv => [B,1,128,128]
        output_image = self.final_conv(up1)
        output_image = self.relu(output_image)
        
        # Resize 128->131
        output_image = self.final_resize(output_image)
        
        return output_image, latent_image

###############################################################
# embedded UnetDecoder model

class UNetDecoder(nn.Module):
    """
    Takes a 2-channel 131x131 image and:
      1) Two-step downsample to 64x64:
         (a) 131->128 by bilinear up/downsampling
         (b) 128->64 by MaxPool2d
      2) Pass through a UNet (3 encoder levels + bottleneck + 3 decoder levels)
      3) Squeeze channels to 1 (1x64x64)
                4) v5: Transformer token heads with structured query design:
                        - patch-token Transformer encoder over the latent map
                        - cross-attention query decoders with:
                            (1) typed scalar queries (MU + jaw1 + jaw2)
                            (2) leaf queries predicted as (prev,curr) pairs (52 tokens per bank)
    """
    def __init__(self, vector_dim, scalar_count):
        super(UNetDecoder, self).__init__()
        
        # ReLU for consistency
        self.relu = nn.ReLU(inplace=True)
        
        # --------------------
        # Step 1: 131 -> 128 (non-trainable bilinear interpolation)
        # Step 2: 128 -> 64  (MaxPool2d, standard downsampling)
        # --------------------
        self.downsample_to_128 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.downsample_128_to_64 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --------------------
        # UNet Encoder (v6: separate prev/curr towers, fuse late)
        # --------------------
        # Each EPID frame is encoded independently as a 1-channel input.
        self.encoder1_prev = EncoderBlock(in_channels=1, out_channels=32)   # 64x64 -> 32x32
        self.encoder2_prev = EncoderBlock(in_channels=32, out_channels=64)  # 32x32 -> 16x16
        self.encoder3_prev = EncoderBlock(in_channels=64, out_channels=128) # 16x16 -> 8x8

        self.encoder1_curr = EncoderBlock(in_channels=1, out_channels=32)   # 64x64 -> 32x32
        self.encoder2_curr = EncoderBlock(in_channels=32, out_channels=64)  # 32x32 -> 16x16
        self.encoder3_curr = EncoderBlock(in_channels=64, out_channels=128) # 16x16 -> 8x8

        # Late fusion (concat + 1x1 conv) for skip features and bottleneck input.
        self.fuse_f1 = nn.Conv2d(32 * 2, 32, kernel_size=1, bias=True)
        self.fuse_f2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=True)
        self.fuse_f3 = nn.Conv2d(128 * 2, 128, kernel_size=1, bias=True)
        self.fuse_p3 = nn.Conv2d(128 * 2, 128, kernel_size=1, bias=True)
        
        # Bottleneck
        self.bottleneck = ResidualConvBlock(in_channels=128, out_channels=256)  # 8x8 (no change in spatial)
        
        # --------------------
        # UNet Decoder
        # --------------------
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128) # 8x8  -> 16x16
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)  # 16x16 -> 32x32
        self.decoder1 = DecoderBlock(in_channels=64,  out_channels=32)  # 32x32 -> 64x64
        
        # --------------------
        # Final conv to get single latent channel
        # --------------------
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # --------------------
        # Transformer heads (replace flatten+FC)
        # --------------------
        self.vector_dim = int(vector_dim)
        self.scalar_count = int(scalar_count)

        self.d_model = 128
        nhead = 4
        ff_dim = 256
        # --------------------
        # Transformer memory (v6: multi-scale tokens)
        # --------------------
        # We build memory tokens from latent_dec at 3 scales:
        # - 64x64 (patch=4 -> 16x16 = 256 tokens)
        # - 32x32 (avgpool2 -> patch=4 -> 8x8 = 64 tokens)
        # - 16x16 (avgpool4 -> patch=4 -> 4x4 = 16 tokens)
        patch = 4
        n64 = (64 // patch) * (64 // patch)
        n32 = (32 // patch) * (32 // patch)
        n16 = (16 // patch) * (16 // patch)

        self.patch_embed_64 = nn.Conv2d(1, self.d_model, kernel_size=patch, stride=patch, bias=True)
        self.patch_embed_32 = nn.Conv2d(1, self.d_model, kernel_size=patch, stride=patch, bias=True)
        self.patch_embed_16 = nn.Conv2d(1, self.d_model, kernel_size=patch, stride=patch, bias=True)

        self.img_pos_embed_64 = nn.Parameter(torch.zeros(1, n64, self.d_model))
        self.img_pos_embed_32 = nn.Parameter(torch.zeros(1, n32, self.d_model))
        self.img_pos_embed_16 = nn.Parameter(torch.zeros(1, n16, self.d_model))

        # Scale embedding lets the transformer disambiguate which scale each token came from.
        self.img_scale_embed = nn.Parameter(torch.zeros(3, self.d_model))

        img_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.img_transformer = nn.TransformerEncoder(img_layer, num_layers=2)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.cross_decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        # --------------------
        # v5: structured queries (typed + prev/curr pairing)
        # --------------------
        if self.vector_dim % 2 != 0:
            raise ValueError(f"vector_dim must be even (prev+curr). Got: {self.vector_dim}")
        self.leaf_bins = self.vector_dim // 2  # 52 when vector_dim=104

        # Leaf queries: 52 base tokens. A bank embedding distinguishes v1 vs v2.
        self.leaf_query_base = nn.Parameter(torch.zeros(1, self.leaf_bins, self.d_model))
        self.bank_embed = nn.Parameter(torch.zeros(2, self.d_model))  # [0]=v1, [1]=v2

        # Typed scalar queries:
        # - MU: one token -> 1 value
        # - Jaws: two tokens (jaw1, jaw2) -> each token -> (prev,curr)
        # Scalar ordering remains: [MU, jaw1_prev, jaw1_curr, jaw2_prev, jaw2_curr]
        self.mu_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.jaw_query = nn.Parameter(torch.zeros(1, 2, self.d_model))

        # Output projections:
        # - Leaves predicted as (prev,curr) pairs for each of the 52 bins per bank
        self.v1_out2 = nn.Linear(self.d_model, 2)
        self.v2_out2 = nn.Linear(self.d_model, 2)
        # - MU predicted as scalar
        self.mu_out = nn.Linear(self.d_model, 1)
        # - Jaws predicted as (prev,curr) pairs per jaw token
        self.jaw_out = nn.Linear(self.d_model, 2)

        nn.init.normal_(self.img_pos_embed_64, std=0.02)
        nn.init.normal_(self.img_pos_embed_32, std=0.02)
        nn.init.normal_(self.img_pos_embed_16, std=0.02)
        nn.init.normal_(self.img_scale_embed, std=0.02)
        nn.init.normal_(self.leaf_query_base, std=0.02)
        nn.init.normal_(self.bank_embed, std=0.02)
        nn.init.normal_(self.mu_query, std=0.02)
        nn.init.normal_(self.jaw_query, std=0.02)

    def forward(self, x):
        """
        x: (batch_size, 2, 131, 131) input image
        Returns:
            reconstructed_vector1, reconstructed_vector2, reconstructed_scalars
        """
        # Some datasets store as (B, 2, 1, H, W). Coerce to (B, 2, H, W).
        if x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)

        # --------------------
        # Two-step downsampling
        # --------------------
        x = self.downsample_to_128(x)       # (B, 2, 128, 128)
        x = self.downsample_128_to_64(x)    # (B, 2, 64, 64)

        # --------------------
        # UNet Encoder (v6: separate prev/curr encoders, fuse late)
        # --------------------
        x_prev = x[:, 0:1, :, :]
        x_curr = x[:, 1:2, :, :]

        f1p, p1p = self.encoder1_prev(x_prev)
        f2p, p2p = self.encoder2_prev(p1p)
        f3p, p3p = self.encoder3_prev(p2p)

        f1c, p1c = self.encoder1_curr(x_curr)
        f2c, p2c = self.encoder2_curr(p1c)
        f3c, p3c = self.encoder3_curr(p2c)

        # Fuse skip features and bottleneck input
        f1 = self.fuse_f1(torch.cat([f1p, f1c], dim=1))
        f2 = self.fuse_f2(torch.cat([f2p, f2c], dim=1))
        f3 = self.fuse_f3(torch.cat([f3p, f3c], dim=1))
        p3 = self.fuse_p3(torch.cat([p3p, p3c], dim=1))
        
        # Bottleneck
        btl = self.bottleneck(p3)   # (B, 256, 8, 8)
        
        # --------------------
        # UNet Decoder
        # --------------------
        u3 = self.decoder3(btl, f3) # (B, 128, 16, 16)
        u2 = self.decoder2(u3, f2)  # (B, 64,  32, 32)
        u1 = self.decoder1(u2, f1)  # (B, 32,  64, 64)
        
        # --------------------
        # Final conv to single channel
        # --------------------
        out = self.final_conv(u1)   # (B, 1, 64, 64)
        out = self.relu(out)        # ReLU activation

        latent_dec = out  # shared latent from images

        # --------------------
        # Patch tokens + transformer memory (v6: multi-scale tokens)
        # --------------------
        # 64x64 tokens
        t64 = self.patch_embed_64(latent_dec)  # (B,d,16,16)
        t64 = t64.flatten(2).transpose(1, 2)   # (B,256,d)
        t64 = t64 + self.img_pos_embed_64[:, :t64.size(1), :] + self.img_scale_embed[0].view(1, 1, -1)

        # 32x32 tokens (avgpool by 2)
        latent_32 = F.avg_pool2d(latent_dec, kernel_size=2, stride=2)
        t32 = self.patch_embed_32(latent_32)   # (B,d,8,8)
        t32 = t32.flatten(2).transpose(1, 2)   # (B,64,d)
        t32 = t32 + self.img_pos_embed_32[:, :t32.size(1), :] + self.img_scale_embed[1].view(1, 1, -1)

        # 16x16 tokens (avgpool by 4)
        latent_16 = F.avg_pool2d(latent_dec, kernel_size=4, stride=4)
        t16 = self.patch_embed_16(latent_16)   # (B,d,4,4)
        t16 = t16.flatten(2).transpose(1, 2)   # (B,16,d)
        t16 = t16 + self.img_pos_embed_16[:, :t16.size(1), :] + self.img_scale_embed[2].view(1, 1, -1)

        tokens = torch.cat([t64, t32, t16], dim=1)  # (B, 256+64+16, d)
        memory = self.img_transformer(tokens)

        B = memory.size(0)

        # --------------------
        # v5 leaf decoding: 52 tokens per bank, each predicts (prev,curr)
        # --------------------
        leaf_base = self.leaf_query_base.expand(B, -1, -1)  # (B,52,d)

        v1_bank = self.bank_embed[0].view(1, 1, -1)
        v2_bank = self.bank_embed[1].view(1, 1, -1)

        v1_tgt = leaf_base + v1_bank
        v2_tgt = leaf_base + v2_bank

        v1_feat = self.cross_decoder(v1_tgt, memory)  # (B,52,d)
        v2_feat = self.cross_decoder(v2_tgt, memory)  # (B,52,d)

        v1_pair = self.v1_out2(v1_feat)  # (B,52,2)
        v2_pair = self.v2_out2(v2_feat)  # (B,52,2)

        # Re-pack to match training target layout: [prev52, curr52] -> 104
        reconstructed_vector1 = torch.cat([v1_pair[:, :, 0], v1_pair[:, :, 1]], dim=1)
        reconstructed_vector2 = torch.cat([v2_pair[:, :, 0], v2_pair[:, :, 1]], dim=1)

        # --------------------
        # v5 scalar decoding: typed MU + jaw1/jaw2 tokens
        # --------------------
        mu_tgt = self.mu_query.expand(B, -1, -1)    # (B,1,d)
        jaw_tgt = self.jaw_query.expand(B, -1, -1)  # (B,2,d)

        mu_feat = self.cross_decoder(mu_tgt, memory)    # (B,1,d)
        jaw_feat = self.cross_decoder(jaw_tgt, memory)  # (B,2,d)

        mu_pred = self.mu_out(mu_feat).squeeze(-1)  # (B,1)
        jaw_pred = self.jaw_out(jaw_feat)           # (B,2,2)
        jaw_flat = jaw_pred.reshape(B, 4)           # (B,4) = [jaw1_prev,jaw1_curr,jaw2_prev,jaw2_curr]

        reconstructed_scalars = torch.cat([mu_pred, jaw_flat], dim=1)  # (B,5)
        
        return reconstructed_vector1, reconstructed_vector2, reconstructed_scalars, latent_dec

#### initalize weights #########################################################

def initialize_weights(m):
    """
    Initialize the weights of layers in a neural network, using typical best practices.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Kaiming (He) normal initialization for convolutional layers
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Linear):
        # Xavier (Glorot) uniform initialization for fully-connected layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # For (Group/Batch/Layer)Norm: weight=1, bias=0 is common
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    
# training_utils.py########################################################

def create_gaussian_kernel(size, sigma, device):
    """
    Create a 2D Gaussian kernel using the specified size and sigma.
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2

    g = coords**2
    g = (-g / (2 * sigma**2)).exp()

    g /= g.sum()
    gaussian_kernel = g[:, None] * g[None, :]
    gaussian_kernel = gaussian_kernel[None, None, :, :]
    return gaussian_kernel

def calculate_gamma_index(ref_data, eval_data, dose_threshold=0.03, distance_mm=3, pixel_spacing=(2.5, 2.5)):
    """
    Calculate the 2D gamma index using PyTorch tensors and return the gamma passing rate.

    :param ref_data: 2D tensor of reference dose distribution.
    :param eval_data: 2D tensor of evaluated dose distribution.
    :param dose_threshold: Dose difference threshold (fraction), typically 0.03 for 3%.
    :param distance_mm: Distance-to-agreement threshold in mm, typically 3 mm.
    :param pixel_spacing: Tuple indicating the pixel spacing in mm (row spacing, column spacing).
    :return: Gamma passing rate as a percentage.
    """
    assert ref_data.shape == eval_data.shape, "Reference and evaluated data must have the same shape"

    max_dose = torch.max(ref_data)
    if max_dose > 0:
        ref_data_normalized = ref_data / max_dose
        eval_data_normalized = eval_data / max_dose
    else:
        return 0

    # Compute dose difference
    dose_diff = torch.abs(ref_data_normalized - eval_data_normalized)

    # Ensure dose_diff is 4D (batched) for F.conv2d
    if dose_diff.dim() == 3:
        dose_diff = dose_diff.unsqueeze(0)  # Add batch dimension if missing

    # Gaussian smoothing for distance-to-agreement
    kernel_size = int(distance_mm / min(pixel_spacing) * 2 + 1)
    sigma = distance_mm / min(pixel_spacing) / 2
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma, ref_data.device)

    # Apply convolution with 'same' padding
    padding_size = (kernel_size - 1) // 2
    distance_agreement = F.conv2d(dose_diff, gaussian_kernel, padding='same')

    # Calculate gamma index
    gamma_index = torch.sqrt((dose_diff / dose_threshold)**2 + (distance_agreement / distance_mm)**2)

    # Calculate gamma passing rate
    gamma_passing_rate = (gamma_index < 1).float().mean().item() * 100

    return gamma_passing_rate

    ##############################################################################

def weighted_mse_loss(input, target, weights):
    squared_error = (input - target) ** 2
    weighted_squared_error = squared_error * weights
    loss = weighted_squared_error.mean()
    return loss

def mse_loss(input, target):
    squared_error = (input - target) ** 2
    loss = squared_error.mean()
    return loss

def weighted_l1_loss(input, target, weights):
    absolute_error = torch.abs(input - target)
    weighted_absolute_error = absolute_error * weights
    return weighted_absolute_error.mean()


def ensure_nchw_epid(x):
    """Coerce EPID tensors to (B, C, H, W).

    Some datasets store EPID images as (B, 1, 1, H, W). The UNet expects 4D.
    """
    if x.dim() == 5 and x.size(2) == 1:
        x = x.squeeze(2)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x


def setup_training(encoderunet, unetdecoder, resume=0):
    base_lr = 1e-4  # Target learning rate after warm-up
    warmup_epochs = 0  # Number of epochs for the warm-up phase
    T_0 = 10  # Epochs for the first cosine restart
    T_mult = 2  # Restart period multiplier
    eta_min = 1e-6  # Minimum learning rate after decay (enable annealing)
    weight_decay = 1e-4

    criterion = nn.MSELoss().to(device)
    scaler = GradScaler()
    
    if resume == 1:
        # Load checkpoint
        checkpoint = torch.load('Cross_CP/EPID_v6_17Dec25_checkpoint.pth', 
                              map_location='cpu')
        
        # Handle state dict for DDP models
        encoderunet_state = {}
        for k, v in checkpoint['encoderunet_state_dict'].items():
            # Remove 'module.' if it exists (from DDP) or add if needed
            if k.startswith('module.'):
                encoderunet_state[k] = v
            else:
                encoderunet_state[f'module.{k}'] = v
                
        unetdecoder_state = {}
        for k, v in checkpoint['unetdecoder_state_dict'].items():
            if k.startswith('module.'):
                unetdecoder_state[k] = v
            else:
                unetdecoder_state[f'module.{k}'] = v
        
        # Load state dicts
        encoderunet.load_state_dict(encoderunet_state)
        unetdecoder.load_state_dict(unetdecoder_state)
        
        # Create optimizer first
        optimizer = AdamW(list(encoderunet.parameters()) + list(unetdecoder.parameters()), 
                         lr=base_lr, weight_decay=weight_decay)
        
        # Load optimizer state dict
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        
        # Update learning rate in optimizer state dict
        for param_group in optimizer_state_dict['param_groups']:
            param_group['lr'] = base_lr
            param_group['initial_lr'] = base_lr
        
        # Move optimizer state to correct device
        for state in optimizer_state_dict['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # Load the processed optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        
        # Scheduler with warm-up + cosine annealing restarts
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: lr_warmup_cosine(epoch, warmup_epochs, base_lr, eta_min, T_0, T_mult)
        )
        
        # Optionally, load scheduler state if you want to maintain the cycle position
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Handle scaler state
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        return (optimizer, scheduler, criterion, start_epoch, 
                train_losses, val_losses, train_accuracies, 
                val_accuracies, scaler)
    else:
        # For encoder
        encoderunet.apply(initialize_weights)
        unetdecoder.apply(initialize_weights)


        optimizer = AdamW(list(encoderunet.parameters()) + list(unetdecoder.parameters()), 
                         lr=base_lr, weight_decay=weight_decay)
        # Scheduler with warm-up + cosine annealing restarts
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: lr_warmup_cosine(epoch, warmup_epochs, base_lr, eta_min, T_0, T_mult)
        )
        start_epoch = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        return (optimizer, scheduler, criterion, start_epoch, 
                train_losses, val_losses, train_accuracies, 
                val_accuracies, scaler)

def lr_warmup_cosine(epoch, warmup_epochs, base_lr, eta_min, T_0, T_mult):
    """
    Combines warm-up with cosine annealing restarts.
    - Warm-up: Gradually increases LR from a small value to base_lr.
    - CosineAnnealingWarmRestarts: Periodically restarts LR using a cosine decay.
    """
    if epoch < warmup_epochs:
        # Linear warm-up phase
        return (epoch + 1) / warmup_epochs
    else:
        # Compute cosine annealing restart phase
        cosine_epochs = epoch - warmup_epochs
        T_cur = T_0
        restart_count = 0

        # Find the current restart cycle
        while cosine_epochs >= T_cur:
            cosine_epochs -= T_cur
            T_cur *= T_mult
            restart_count += 1

        # Convert to tensor for cosine calculation
        cosine_input = torch.tensor(cosine_epochs / T_cur * math.pi)
        # Compute the learning rate scale for the current cycle
        cosine_decay = 0.5 * (1 + torch.cos(cosine_input))
        return eta_min / base_lr + (1 - eta_min / base_lr) * cosine_decay.item()


# training_loop.py

def train_cross(encoderunet, unetdecoder, train_loaders, val_loaders, device, batch_size, resume=0):
    # Get training setup (now returns scaler instead of scaler_state)
    optimizer, scheduler, criterion, start_epoch, train_losses, val_losses, train_accuracies, val_accuracies, scaler = setup_training(encoderunet, unetdecoder, resume)

    # Remove this line since scaler is now directly returned from setup_training
    # if scaler_state is not None:
    #     scaler.load_state_dict(scaler_state)

    # Get rank for printing
    rank = dist.get_rank()
    
    # Setup training parameters
    EPOCHS = 2000

    # Print settings
    print('SETTINGS')
    print('epochs:', EPOCHS)
    print('batch size:', batch_size)
    print('optimizer: Adam')
    print('learning rate:', optimizer.param_groups[0]['lr'])
    print('loss: weighted L1 + MSE')
    print('using mixed precision training')

    sys.stdout.flush()
    
    line_length = 155
    
    for epoch in range(start_epoch, EPOCHS):
        # Synchronize GPUs before starting epoch
        #torch.cuda.synchronize()
        #dist.barrier()

        # Training
        encoderunet.train()
        unetdecoder.train()
        
        running_train_losses = [0.0] * len(train_loaders)
        running_train_accuracies = [0.0] * len(train_loaders)
        
        start_time = time.time()
        
        for i, train_loader in enumerate(train_loaders):
            loader_loss_sum = 0.0
            loader_accuracy_sum = 0.0
            
            for batch_idx, (v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p) in enumerate(train_loader):
                # Move data to device
                v1, v2, scalars = v1.to(device), v2.to(device), scalars.to(device)
                v1_weight, v2_weight = v1_weight.to(device), v2_weight.to(device)
                arrays, arrays_p = arrays.to(device), arrays_p.to(device)

                arrays = ensure_nchw_epid(arrays)
                arrays_p = ensure_nchw_epid(arrays_p)
                
                # Wrap training steps with autocast
                with autocast():
                    # Forward pass through encoder-unet
                    outputs, latent_enc = encoderunet(v1, v2, scalars)
                    accuracy = 0
                    
                    # Compute "forward loss"
                    mu = scalars[:, 0, 0].clamp(min=1e-6)  # [B]
                    weights = (1.0 / mu).view(-1, 1, 1, 1)
                    loss_for = weighted_mse_loss(outputs, arrays, weights)
                    
                    # First decoder pass
                    arrays_con = torch.cat([arrays_p, arrays], dim=1)
                    v1_reconstructed, v2_reconstructed, scalars_reconstructed, latent_dec = unetdecoder(arrays_con)
                    
                    # Prepare tensors
                    v1, v2 = v1.squeeze(1), v2.squeeze(1)
                    v1_weight, v2_weight = v1_weight.squeeze(1), v2_weight.squeeze(1)
                    scalars = scalars.squeeze(1).squeeze(-1)

                    # Normalize vectors/scalars to keep losses numerically balanced
                    norm_vec = 130.0
                    scalar_scale = scalars.new_tensor([40.0, 130.0, 130.0, 130.0, 130.0]).view(1, 5)
                    v1_pred = v1_reconstructed / norm_vec
                    v2_pred = v2_reconstructed / norm_vec
                    v1_tgt = v1 / norm_vec
                    v2_tgt = v2 / norm_vec
                    scalars_pred = scalars_reconstructed / scalar_scale
                    scalars_tgt = scalars / scalar_scale

                    # Penalty losses (on normalized values)
                    v1_ref = v1_pred.detach()
                    penalty_loss = torch.relu(v1_ref - v2_pred).pow(2).mean()
                    
                    # Consistency losses
                    consistency_loss = 0.0
                    if v1_reconstructed.size(0) > 1:
                        mse_loss_v1_diff = weighted_mse_loss(
                            v1_pred[:-1, -52:],
                            v1_pred[1:, :52],
                            v1_weight[:-1, -52:]
                        )
                        mse_loss_v2_diff = weighted_mse_loss(
                            v2_pred[:-1, -52:],
                            v2_pred[1:, :52],
                            v2_weight[:-1, -52:]
                        )
                        
                        consistency_loss = (mse_loss_v1_diff + mse_loss_v2_diff)
                    
                    # Reconstruction losses
                    mse_loss_v1 = weighted_mse_loss(v1_pred, v1_tgt, v1_weight)
                    mse_loss_v2 = weighted_mse_loss(v2_pred, v2_tgt, v2_weight)
                    mse_loss_scalars = criterion(scalars_pred, scalars_tgt)
                    latent_loss = mse_loss(latent_enc, latent_dec)
                    
                    loss_back = (mse_loss_v1 + mse_loss_v2 + mse_loss_scalars +
                                penalty_loss * 10)
                    
                    # Total loss
                    loss = loss_for + loss_back + consistency_loss + 0.1 * latent_loss
                
                # Optimization step with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Synchronize after optimization
                #torch.cuda.synchronize()
                #dist.barrier()
                
                loader_loss_sum += loss.item()
                loader_accuracy_sum += accuracy

                if rank == 0 and i in [0, 80, 160, 240] and batch_idx in [5]:
                    cons_val = consistency_loss if isinstance(consistency_loss, float) else consistency_loss.item()
                    progress_msg = (
                        f"Epoch [{epoch+1}/{EPOCHS}] Loader [{i+1}/{len(train_loaders)}] "
                        f"Batch [{batch_idx+1}/{len(train_loader)}]\n"
                        f"Train Loss Components: for={loss_for.item():.2e}, back={loss_back.item():.2e}, "
                        f"cons={cons_val:.2e}, latent={latent_loss.item():.2e}, total={loss.item():.2e}\n"
                        f"MSE Loss Components: v1={mse_loss_v1.item():.2e}, v2={mse_loss_v2.item():.2e}, "
                        f"scl={mse_loss_scalars.item():.2e}, Penalty Losses: pen1={penalty_loss.item():.2e}"
                    )
                    print(progress_msg.ljust(300), flush=True)  # Adjust width as needed
            
            # Calculate averages for this loader
            num_batches = len(train_loader)
            running_train_losses[i] = loader_loss_sum / num_batches
            running_train_accuracies[i] = loader_accuracy_sum / num_batches

            


        # Validation
        encoderunet.eval()
        unetdecoder.eval()
        
        with torch.no_grad(), autocast():
            running_val_losses = [0.0] * len(val_loaders)
            running_val_accuracies = [0.0] * len(val_loaders)
            
            for i, val_loader in enumerate(val_loaders):
                loader_loss_sum = 0.0
                loader_accuracy_sum = 0.0
                
                for batch_idx, (v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p) in enumerate(val_loader):
                    # Move data to device
                    v1, v2, scalars = v1.to(device), v2.to(device), scalars.to(device)
                    v1_weight, v2_weight = v1_weight.to(device), v2_weight.to(device)
                    arrays, arrays_p = arrays.to(device), arrays_p.to(device)

                    arrays = ensure_nchw_epid(arrays)
                    arrays_p = ensure_nchw_epid(arrays_p)
                    
                    # Forward pass through encoder-unet
                    outputs, latent_enc = encoderunet(v1, v2, scalars)
                    #accuracy = calculate_gamma_index(outputs, arrays)
                    accuracy = 0

                    # Compute "forward loss"
                    mu = scalars[:, 0, 0].clamp(min=1e-6)  # [B]
                    weights = (1.0 / mu).view(-1, 1, 1, 1)
                    loss_for = weighted_mse_loss(outputs, arrays, weights)
                    
                    # First decoder pass
                    arrays_con = torch.cat([arrays_p, arrays], dim=1)
                    v1_reconstructed, v2_reconstructed, scalars_reconstructed, latent_dec = unetdecoder(arrays_con)
                    
                    # Prepare tensors
                    v1, v2 = v1.squeeze(1), v2.squeeze(1)
                    v1_weight, v2_weight = v1_weight.squeeze(1), v2_weight.squeeze(1)
                    scalars = scalars.squeeze(1).squeeze(-1)

                    # Normalize vectors/scalars for balanced losses
                    norm_vec = 130.0
                    scalar_scale = scalars.new_tensor([40.0, 130.0, 130.0, 130.0, 130.0]).view(1, 5)
                    v1_pred = v1_reconstructed / norm_vec
                    v2_pred = v2_reconstructed / norm_vec
                    v1_tgt = v1 / norm_vec
                    v2_tgt = v2 / norm_vec
                    scalars_pred = scalars_reconstructed / scalar_scale
                    scalars_tgt = scalars / scalar_scale

                    # Penalty losses (on normalized values)
                    v1_ref = v1_pred.detach()
                    penalty_loss = torch.relu(v1_ref - v2_pred).pow(2).mean()
                    
                    # Consistency losses
                    consistency_loss = 0.0
                    if v1_reconstructed.size(0) > 1:
                        mse_loss_v1_diff = weighted_mse_loss(
                            v1_pred[:-1, -52:],
                            v1_pred[1:, :52],
                            v1_weight[:-1, -52:]
                        )
                        mse_loss_v2_diff = weighted_mse_loss(
                            v2_pred[:-1, -52:],
                            v2_pred[1:, :52],
                            v2_weight[:-1, -52:]
                        )
                        
                        consistency_loss = (mse_loss_v1_diff + mse_loss_v2_diff)
                    
                    # Reconstruction losses
                    mse_loss_v1 = weighted_mse_loss(v1_pred, v1_tgt, v1_weight)
                    mse_loss_v2 = weighted_mse_loss(v2_pred, v2_tgt, v2_weight)
                    mse_loss_scalars = criterion(scalars_pred, scalars_tgt)
                    latent_loss = mse_loss(latent_enc, latent_dec)
                    
                    loss_back = (mse_loss_v1 + mse_loss_v2 + mse_loss_scalars +
                                penalty_loss * 10)
                    
                    # Total loss
                    loss = loss_for + loss_back + consistency_loss + 0.1 * latent_loss
                    
                    loader_loss_sum += loss.item()
                    loader_accuracy_sum += accuracy
                
                # Calculate averages for this loader
                num_batches = len(val_loader)
                running_val_losses[i] = loader_loss_sum / num_batches
                running_val_accuracies[i] = loader_accuracy_sum / num_batches

                #print(f"Epoch [{epoch+1}/{EPOCHS}] Loader [{i+1}/{len(val_loaders)}] "
                #      f"Batch [{batch_idx+1}/{len(val_loader)}]  "
                #      f"Temp. Val. Loss: {loss_for.item():.2e} {loss_back.item():.2e} "
                #      f"{loss_for_2.item():.2e} {loss_back_2.item():.2e} {consistency_loss.item():.2e} "
                #      f"{loss.item():.2e}  Temp. Val. Acc.: {accuracy:.2f}".ljust(line_length), end='\r')


             # Calculate average metrics for the epoch
        average_train_loss = sum(running_train_losses) / len(train_loaders)
        average_train_accuracy = sum(running_train_accuracies) / len(train_loaders)
        average_val_loss = sum(running_val_losses) / len(val_loaders)
        average_val_accuracy = sum(running_val_accuracies) / len(val_loaders)

        # Append to history
        #train_losses.append(average_train_loss)
        #train_accuracies.append(average_train_accuracy)
        #val_losses.append(average_val_loss)
        #val_accuracies.append(average_val_accuracy)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Gather losses and accuracies from all GPUs
        world_size = dist.get_world_size()
        all_train_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
        all_train_accuracies = [torch.zeros(1).to(device) for _ in range(world_size)]
        all_val_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
        all_val_accuracies = [torch.zeros(1).to(device) for _ in range(world_size)]
        
        # Convert local values to tensors
        local_train_loss = torch.tensor([average_train_loss]).to(device)
        local_train_acc = torch.tensor([average_train_accuracy]).to(device)
        local_val_loss = torch.tensor([average_val_loss]).to(device)
        local_val_acc = torch.tensor([average_val_accuracy]).to(device)
        
        # Gather from all GPUs
        dist.all_gather(all_train_losses, local_train_loss)
        dist.all_gather(all_train_accuracies, local_train_acc)
        dist.all_gather(all_val_losses, local_val_loss)
        dist.all_gather(all_val_accuracies, local_val_acc)

        # Calculate global averages
        global_train_loss = sum([loss.item() for loss in all_train_losses]) / world_size
        global_train_acc = sum([acc.item() for acc in all_train_accuracies]) / world_size
        global_val_loss = sum([loss.item() for loss in all_val_losses]) / world_size
        global_val_acc = sum([acc.item() for acc in all_val_accuracies]) / world_size

        # Print epoch summary with global averages
        if rank == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Avg. Train Loss: {global_train_loss:.4e} "
                  f"Avg. Train Accuracy: {global_train_acc:.2f} "
                  f"Avg. Val. Loss: {global_val_loss:.4e} "
                  f"Avg. Val. Accuracy: {global_val_acc:.2f} "
                  f"Elap. Time: {elapsed_time:.1f} seconds "
                  f"Current LR: {current_lr:.4e}")
            
            sys.stdout.flush()

        # Store global averages
        train_losses.append(global_train_loss)
        train_accuracies.append(global_train_acc)
        val_losses.append(global_val_loss)
        val_accuracies.append(global_val_acc)

        # Only save checkpoint from rank 0
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'encoderunet_state_dict': encoderunet.module.state_dict(),
                'unetdecoder_state_dict': unetdecoder.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }
            
            torch.save(checkpoint, 'Cross_CP/EPID_v6_17Dec25_checkpoint.pth')


    return train_losses, val_losses, train_accuracies, val_accuracies           

    #######################################################################




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        "nccl", 
        #"gloo", #local
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=60)
    )
    
    # Set device and CUDA settings
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False    



def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, generate_flag, KM):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    train_loaders = []
    val_loaders = []

    # Calculate dataset range for this rank

    datasets_per_gpu = 640 // world_size
    start_dataset = rank * datasets_per_gpu
    end_dataset = start_dataset + datasets_per_gpu

    # Generate or load datasets assigned to this GPU
    for dataset_num in range(start_dataset, end_dataset):
    #for dataset_num in range(0, 640, 80): #local
        start_time = time.time()

        if generate_flag == 0:
            Art_dataset = load_dataset(dataset_num)
            if Art_dataset is None:
                Art_dataset = generate_and_save_dataset(dataset_num, KM)
                print(f"[GPU {rank}] Generated and saved dataset {dataset_num} because it was not found on disk")
            else:
                print(f"[GPU {rank}] Loaded dataset {dataset_num} from disk")
        else:
            Art_dataset = generate_and_save_dataset(dataset_num, KM)
            print(f"[GPU {rank}] Generated and saved dataset {dataset_num}")

        sys.stdout.flush()  # Ensure prints are flushed immediately
        
        # Split into train and validation sets (sequential split, no randomization)
        VALIDSPLIT = 0.8125
        dataset_size = len(Art_dataset)
        split = int(np.floor(VALIDSPLIT * dataset_size))
        
        # Sequential split
        train_indices = list(range(split))
        val_indices = list(range(split, dataset_size))

        train_ds = Subset(Art_dataset, train_indices)
        val_ds = Subset(Art_dataset, val_indices)

        # Adjust batch size based on available GPU memory
        batch_size = 128// world_size  # Scale batch size by number of GPUs
    

        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size,
            shuffle=False,  # Keep sequential order
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        end_time = time.time()
        print(f"[GPU {rank}] Dataset {dataset_num} processing time: {end_time - start_time:.2f} seconds")
        print(f"[GPU {rank}] Dataset {dataset_num} is done")

    # Initialize models
    vector_dim = 104
    scalar_count = 5


    encoderunet = EncoderUNet(vector_dim, scalar_count)
    encoderunet = encoderunet.to(device)
    encoderunet = DDP(encoderunet, device_ids=[rank])

 
    unetdecoder = UNetDecoder(vector_dim, scalar_count)

    unetdecoder = unetdecoder.to(device)
    unetdecoder = DDP(unetdecoder, device_ids=[rank])

    if rank == 0:
        print("\nModel devices:")
        print(f"encoderunet device: {next(encoderunet.parameters()).device}")
        print(f"unetdecoder device: {next(unetdecoder.parameters()).device}")
        print("Starting training...")
    
    train_cross(encoderunet, unetdecoder, train_loaders, val_loaders, device, batch_size, resume=0)
    
    if rank == 0:
        print("Training completed!")
    
    cleanup()

if __name__ == "__main__":
    # Load KM matrix
    KM_data = scipy.io.loadmat('data/KM_1500.mat')
    KM = KM_data['KM_1500']

    # Create directories if they don't exist
    os.makedirs("VMAT_Art_data", exist_ok=True)
    os.makedirs("Cross_CP", exist_ok=True)

    # Set generation flag
    generate_flag = 0  # Set this flag to 1 if you want to generate datasets again

    # Launch training on 2 GPUs
    world_size = 1
    mp.spawn(
        train_ddp,
        args=(world_size, generate_flag, KM),
        nprocs=world_size,
        join=True
    )