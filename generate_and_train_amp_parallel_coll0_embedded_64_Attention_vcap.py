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


################################################################################
# 0. Attention Gating (for skip connections)
################################################################################
class AttentionGate(nn.Module):
    """
    Implements a simple attention gating mechanism (similar to 'Attention U-Net'):
      - Gating (decoder) feature (g) -> 1x1 Conv -> transforms to match skip dimension
      - Skip (x) feature -> 1x1 Conv
      - Sum + ReLU -> 1x1 Conv -> sigmoid => gating mask alpha
      - Output = x * alpha (elementwise scaling)

    Args:
      in_channels:  channels of the skip feature
      gating_channels: channels of the decoder feature

    References:
      Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
      https://arxiv.org/abs/1804.03999
    """
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()
        
        self.Wx = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.Wg = nn.Conv2d(gating_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, g):
        """
        x: Skip feature map  (B, in_channels, H, W)
        g: Gating feature map (B, gating_channels, H, W) 
           -> typically from the decoder's upsampled output
        """
        # 1) Project skip & gating to the same intermediate shape
        x_proj = self.Wx(x)  # (B, inC, H, W)
        g_proj = self.Wg(g)  # (B, inC, H, W)  (assuming gating_channels≥inC or are projected down)
        
        # 2) Add + ReLU
        combined = self.relu(x_proj + g_proj)
        
        # 3) Channel squeeze to 1 channel => sigmoid => alpha
        alpha = self.psi(combined)  # (B, 1, H, W)
        alpha = self.sigmoid(alpha) # (B, 1, H, W)
        
        # 4) Scale skip feature
        out = x * alpha
        return out

################################################################################
# 1. TransformerBottleneck (same as before)
################################################################################
class TransformerBottleneck(nn.Module):
    """
    A simple Transformer-based bottleneck:
      1) Conv1x1 to project in_channels -> mid_channels
      2) Flatten (H*W) as 'tokens', apply TransformerEncoderLayer
      3) Reshape back, Conv1x1, ReLU
    """
    def __init__(self, in_channels=128, mid_channels=256, n_heads=4, dim_feedforward=1024):
        super().__init__()
        
        self.proj_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=mid_channels, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        
        self.proj_out = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        return: (B, mid_channels, H, W)
        """
        B, C, H, W = x.size()
        
        # 1) Project
        x = self.proj_in(x)         # (B, midC, H, W)
        
        # 2) Flatten for Transformer: (B, H*W, midC)
        x = x.flatten(2)            # (B, midC, H*W)
        x = x.permute(0, 2, 1)      # (B, H*W, midC) -> batch_first = True
        
        # 3) Self-attention
        x = self.transformer(x)     # (B, H*W, midC)
        
        # 4) Reshape
        x = x.permute(0, 2, 1)      # (B, midC, H*W)
        x = x.view(B, -1, H, W)     # (B, midC, H, W)
        
        # 5) Final conv + ReLU
        x = self.proj_out(x)
        x = self.relu(x)
        return x

################################################################################
# 2. ResidualConvBlock
################################################################################
class ResidualConvBlock(nn.Module):
    """
    A 2D residual block with:
      - two conv layers (3x3)
      - GroupNorm(1, C) after each conv
      - ReLU
      - optional 1x1 conv shortcut if channels mismatch
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(1, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.relu(out)
        return out

################################################################################
# 3. EncoderBlock
################################################################################
class EncoderBlock(nn.Module):
    """
    1) ResidualConvBlock
    2) MaxPool2d
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.conv_block = ResidualConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f = self.conv_block(x)
        p = self.pool(f)
        return f, p

################################################################################
# 4. AttentionDecoderBlock (replaces standard DecoderBlock)
################################################################################
class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with an additional AttentionGate for the skip connection.

    Steps:
      1) ConvTranspose2d (inC->outC) to upsample
      2) Pass the skip features + upsampled feature map to an AttentionGate
         => skip_features = att_gate(skip_features, upsampled_x)
      3) Concatenate skip & upsampled => ResidualConvBlock => out
    """
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super(AttentionDecoderBlock, self).__init__()
        
        if skip_channels is None:
            skip_channels = out_channels  # typical UNet pattern

        # 2x upsampling
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Attention gate for skip
        self.att_gate = AttentionGate(in_channels=skip_channels, gating_channels=out_channels)
        
        # After gating, we still concatenate: out_channels (upsampled) + skip_channels
        self.conv_block = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_features):
        """
        x: (B, inC, H, W)
        skip_features: (B, skip_channels, H, W)
        """
        # 1) Upsample
        x_up = self.conv_transpose(x)  # => (B, outC, 2H, 2W)
        
        # 2) Gate skip
        gated_skip = self.att_gate(skip_features, x_up)  # => (B, skipC, 2H, 2W)
        
        # 3) Concat + residual conv
        x_cat = torch.cat([x_up, gated_skip], dim=1)  # => (B, outC+skipC, 2H, 2W)
        out = self.conv_block(x_cat)
        return out

################################################################################
# 5. EncoderUNet with Transformer Bottleneck + Attention in Skips
################################################################################
class EncoderUNet(nn.Module):
    """
    UNet-like architecture:
      - 3-level encoder
      - Transformer bottleneck
      - 3-level decoder with AttentionGate skip connections
      - Upsample to 128x128, final conv => (B,1,128,128), then resize to (131,131)

    Also includes:
      - Vector+Scalar processing => latent image
    """
    def __init__(self, vector_dim, scalar_count):
        super(EncoderUNet, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Vector processing
        self.vector_fc = nn.Linear(vector_dim * 2, 128)
        self.vector_norm = nn.LayerNorm(128)
        
        # Scalar processing
        self.scalar_fc_list = nn.ModuleList([
            nn.Linear(1, 16) for _ in range(scalar_count)
        ])
        self.scalar_norm = nn.LayerNorm(scalar_count * 16)
        
        # Combine => latent_to_image => (B,1,64,64)
        in_features = 128 + scalar_count * 16
        self.latent_to_image = nn.Linear(in_features, 64*64)
        
        # Encoder
        self.encoder1 = EncoderBlock(1,   32)   # 64->32
        self.encoder2 = EncoderBlock(32,  64)   # 32->16
        self.encoder3 = EncoderBlock(64,  128)  # 16->8
        
        # Transformer bottleneck
        self.bottleneck = TransformerBottleneck(
            in_channels=128,
            mid_channels=256,
            n_heads=4,
            dim_feedforward=1024
        )
        
        # Decoder (Attention-based)
        self.decoder3 = AttentionDecoderBlock(in_channels=256, out_channels=128, skip_channels=128) # 8->16
        self.decoder2 = AttentionDecoderBlock(in_channels=128, out_channels=64,  skip_channels=64)  # 16->32
        self.decoder1 = AttentionDecoderBlock(in_channels=64,  out_channels=32,  skip_channels=32)  # 32->64
        
        # Upsample 64->128
        self.up1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        # Final conv => (B,1,128,128), then resize to 131
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.final_resize = nn.Upsample(size=(131,131), mode='bilinear', align_corners=False)

    def forward(self, vector1, vector2, scalars):
        B = vector1.size(0)
        
        # Flatten & combine vectors
        vector1 = vector1.view(B, -1)
        vector2 = vector2.view(B, -1)
        vec_input = torch.cat([vector1, vector2], dim=1)  # => (B, 2*vector_dim)
        
        vec_features = self.vector_fc(vec_input)
        vec_features = self.vector_norm(vec_features)
        vec_features = self.relu(vec_features)
        
        # Process scalars
        if scalars.dim() == 3 and scalars.size(1) == 1:
            scalars = scalars.squeeze(1)  # => (B, scalar_count)
        
        scalar_list = []
        for i, fc_layer in enumerate(self.scalar_fc_list):
            s_i = scalars[:, i].unsqueeze(1)
            s_i_emb = fc_layer(s_i)      # => (B,16)
            scalar_list.append(s_i_emb)
        scalar_features = torch.cat(scalar_list, dim=1)
        scalar_features = self.scalar_norm(scalar_features)
        scalar_features = self.relu(scalar_features)
        
        combined = torch.cat([vec_features, scalar_features], dim=1) # => (B,128 +16*scalar_count)
        latent = self.latent_to_image(combined)                     # => (B, 4096)
        latent = latent.view(B,1,64,64)                             # => (B,1,64,64)
        
        # Encoder
        f1, p1 = self.encoder1(latent)  # 64->32
        f2, p2 = self.encoder2(p1)      # 32->16
        f3, p3 = self.encoder3(p2)      # 16->8
        
        # Transformer Bottleneck
        btl = self.bottleneck(p3)       # 8->8 (256 channels)
        
        # Decoder (with attention gating)
        u3 = self.decoder3(btl, f3)     # 8->16
        u2 = self.decoder2(u3, f2)      # 16->32
        u1 = self.decoder1(u2, f1)      # 32->64
        
        # Upsample 64->128
        up1 = self.up1(u1)             # => (B,16,128,128)
        up1 = self.relu(up1)
        
        # Final conv -> (B,1,128,128)
        out = self.final_conv(up1)
        out = self.relu(out)
        
        # Resize to (131,131)
        out = self.final_resize(out)
        
        return out

################################################################################
# 6. UNetDecoder with Transformer + Attention Gating in Skips
################################################################################
class UNetDecoder(nn.Module):
    """
    Takes a 2-channel 131x131 image and:
      1) Downsamples to 64x64
      2) UNet (3-level encoder + Transformer bottleneck + 3-level attention decoder)
      3) Final -> (B,1,64,64), flatten => produce vectors/scalars
    """
    def __init__(self, vector_dim, scalar_count):
        super(UNetDecoder, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample steps
        self.downsample_to_128 = nn.Upsample(size=(128,128), mode='bilinear', align_corners=False)
        self.downsample_128_to_64 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.encoder1 = EncoderBlock(in_channels=2,  out_channels=32) # 64->32
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64) # 32->16
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128)# 16->8
        
        # Transformer Bottleneck
        self.bottleneck = TransformerBottleneck(
            in_channels=128,
            mid_channels=256,
            n_heads=4,
            dim_feedforward=1024
        )
        
        # Attention-based Decoder
        self.decoder3 = AttentionDecoderBlock(in_channels=256, out_channels=128, skip_channels=128) # 8->16
        self.decoder2 = AttentionDecoderBlock(in_channels=128, out_channels=64,  skip_channels=64)  # 16->32
        self.decoder1 = AttentionDecoderBlock(in_channels=64,  out_channels=32,  skip_channels=32)  # 32->64
        
        # Final conv => single latent channel (1, 64, 64)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # Flatten + FC for output
        self.fc_main = nn.Linear(64*64, 512)
        self.fc_norm = nn.LayerNorm(512)

        # (A) **Extra capacity** for vector1
        self.vector_mlp1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, vector_dim)  # e.g. 104
        )
        
        # (B) **Extra capacity** for vector2
        self.vector_mlp2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, vector_dim)
        )
        
    
        self.scalar_fc  = nn.Linear(512, scalar_count)

    def forward(self, x):
        """
        x: (B,2,131,131)
        Returns:
          reconstructed_vector1, reconstructed_vector2, reconstructed_scalars
        """
        # Downsample
        x = self.downsample_to_128(x)    # (B,2,128,128)
        x = self.downsample_128_to_64(x) # (B,2,64,64)
        
        # Encoder
        f1, p1 = self.encoder1(x)  # => 32->32
        f2, p2 = self.encoder2(p1) # => 64->16
        f3, p3 = self.encoder3(p2) # => 128->8
        
        # Bottleneck
        btl = self.bottleneck(p3)  # => (B,256,8,8)
        
        # Decoder
        u3 = self.decoder3(btl, f3)  # => 16->16
        u2 = self.decoder2(u3, f2)   # => 32->32
        u1 = self.decoder1(u2, f1)   # => 64->64
        
        # Final conv => (B,1,64,64)
        out = self.final_conv(u1)
        out = self.relu(out)
        
        # Flatten => FC => separate heads
        out = out.view(out.size(0), -1)   # (B,4096)
        out = self.fc_main(out)          # (B,512)
        out = self.fc_norm(out)
        out = self.relu(out)
        
        # (A) Reconstruct vector1
        vector1 = self.vector_mlp1(out)  # => (B, vector_dim)
        
        # (B) Reconstruct vector2
        vector2 = self.vector_mlp2(out)  # => (B, vector_dim)

        scalars = self.scalar_fc(out)    # (B, scalar_count)
        
        return vector1, vector2, scalars

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

def weighted_l1_loss(input, target, weights):
    absolute_error = torch.abs(input - target)
    weighted_absolute_error = absolute_error * weights
    return weighted_absolute_error.mean()


def setup_training(encoderunet, unetdecoder, resume=0):
    base_lr = 1e-4  # Target learning rate after warm-up
    warmup_epochs = 0  # Number of epochs for the warm-up phase
    T_0 = 10  # Epochs for the first cosine restart
    T_mult = 2  # Restart period multiplier
    eta_min = 1e-4  # Minimum learning rate after decay
    weight_decay = 1e-4

    criterion = nn.MSELoss().to(device)
    scaler = GradScaler()
    
    if resume == 1:
        # Load checkpoint
        checkpoint = torch.load('Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_embedded_64_Attention_vcap_checkpoint.pth', 
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
    EPOCHS = 600

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
                
                arrays = arrays.squeeze(1)
                arrays_p = arrays_p.squeeze(1)
                
                # Wrap training steps with autocast
                with autocast():
                    # Forward pass through encoder-unet
                    outputs = encoderunet(v1, v2, scalars)
                    accuracy = 0
                    
                    # Compute "forward loss"
                    l1_loss_per_element = F.l1_loss(outputs, arrays, reduction='none')
                    l1_loss_per_sample = l1_loss_per_element.sum(dim=[2, 3]).squeeze()
                    weight = 1/scalars[:,0,0]
                    loss_for = (l1_loss_per_sample * weight).mean()
                    
                    # First decoder pass
                    arrays_con = torch.cat([arrays_p, arrays], dim=1)
                    v1_reconstructed, v2_reconstructed, scalars_reconstructed = unetdecoder(arrays_con)
                    
                    # Second encoder pass
                    outputs_2 = encoderunet(v1_reconstructed.unsqueeze(1), 
                                          v2_reconstructed.unsqueeze(1), 
                                          scalars_reconstructed.unsqueeze(1))
                    
                    # Compute "second pass forward loss"
                    l1_loss_per_element = F.l1_loss(outputs_2, arrays, reduction='none')
                    l1_loss_per_sample = l1_loss_per_element.sum(dim=[2, 3]).squeeze()
                    weight = 1/scalars_reconstructed.unsqueeze(1)[:,0,0]
                    loss_for_2 = (l1_loss_per_sample * weight).mean()
                    
                    # Second decoder pass
                    arrays_p = outputs[:-1]
                    main_batch = outputs[1:]
                    arrays_con_2 = torch.cat([arrays_p, main_batch], dim=1)
                    v1_reconstructed_2, v2_reconstructed_2, scalars_reconstructed_2 = unetdecoder(arrays_con_2)
                    
                    # Prepare tensors
                    v1, v2 = v1.squeeze(1), v2.squeeze(1)
                    v1_weight, v2_weight = v1_weight.squeeze(1), v2_weight.squeeze(1)
                    scalars = scalars.squeeze(1)
                    
                    # Penalty losses
                    penalty_loss = torch.where(v2_reconstructed < v1_reconstructed,
                                             v1_reconstructed - v2_reconstructed,
                                             torch.zeros_like(v2_reconstructed)).sum()
                    
                    penalty_loss_2 = torch.where(v2_reconstructed_2 < v1_reconstructed_2,
                                               v1_reconstructed_2 - v2_reconstructed_2,
                                               torch.zeros_like(v2_reconstructed_2)).sum()
                    
                    # Consistency losses
                    if v1_reconstructed.size(0) > 1:
                        mse_loss_v1_diff = weighted_l1_loss(
                            v1_reconstructed[:-1, -52:],
                            v1_reconstructed[1:, :52],
                            v1_weight[:-1, -52:]
                        )
                        mse_loss_v2_diff = weighted_l1_loss(
                            v2_reconstructed[:-1, -52:],
                            v2_reconstructed[1:, :52],
                            v2_weight[:-1, -52:]
                        )
                        mse_loss_v1_diff_2 = weighted_l1_loss(
                            v1_reconstructed_2[:-1, -52:],
                            v1_reconstructed_2[1:, :52],
                            v1_weight[1:-1, -52:]
                        )
                        mse_loss_v2_diff_2 = weighted_l1_loss(
                            v2_reconstructed_2[:-1, -52:],
                            v2_reconstructed_2[1:, :52],
                            v2_weight[1:-1, -52:]
                        )
                        
                        consistency_loss = (mse_loss_v1_diff + mse_loss_v2_diff +
                                          mse_loss_v1_diff_2 + mse_loss_v2_diff_2)
                    
                    # Reconstruction losses
                    mse_loss_v1 = weighted_l1_loss(v1_reconstructed, v1, v1_weight)
                    mse_loss_v2 = weighted_l1_loss(v2_reconstructed, v2, v2_weight)
                    mse_loss_scalars = criterion(scalars_reconstructed, scalars) * 5
                    
                    loss_back = (mse_loss_v1 + mse_loss_v2 + mse_loss_scalars +
                                penalty_loss * 10)
                    
                    mse_loss_v1_2 = weighted_l1_loss(v1_reconstructed_2, v1[1:], v1_weight[1:])
                    mse_loss_v2_2 = weighted_l1_loss(v2_reconstructed_2, v2[1:], v2_weight[1:])
                    mse_loss_scalars_2 = criterion(scalars_reconstructed_2, scalars[1:]) * 5
                    
                    loss_back_2 = (mse_loss_v1_2 + mse_loss_v2_2 + mse_loss_scalars_2 +
                                  penalty_loss_2 * 10)
                    
                    # Total loss
                    loss = loss_for + loss_back + loss_for_2 + loss_back_2 + consistency_loss
                
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
            
            # Calculate averages for this loader
            num_batches = len(train_loader)
            running_train_losses[i] = loader_loss_sum / num_batches
            running_train_accuracies[i] = loader_accuracy_sum / num_batches

            # Print progress
            #print(f"Epoch [{epoch+1}/{EPOCHS}] Loader [{i+1}/{len(train_loaders)}] "
            #      f"Batch [{batch_idx+1}/{len(train_loader)}]  "
            #      f"Temp. Train. Loss: {loss_for.item():.2e} {loss_back.item():.2e} "
            #      f"{loss_for_2.item():.2e} {loss_back_2.item():.2e} {consistency_loss.item():.2e} "
            #      f"{loss.item():.2e}  Temp. Train. Acc.: {accuracy:.2f}".ljust(line_length), end='\r')


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
                    
                    arrays = arrays.squeeze(1)
                    arrays_p = arrays_p.squeeze(1)
                    
                    # Forward pass through encoder-unet
                    outputs = encoderunet(v1, v2, scalars)
                    #accuracy = calculate_gamma_index(outputs, arrays)
                    accuracy = 0

                    # Compute "forward loss"
                    l1_loss_per_element = F.l1_loss(outputs, arrays, reduction='none')
                    l1_loss_per_sample = l1_loss_per_element.sum(dim=[2, 3]).squeeze()
                    weight = 1/scalars[:,0,0]
                    loss_for = (l1_loss_per_sample * weight).mean()
                    
                    # First decoder pass
                    arrays_con = torch.cat([arrays_p, arrays], dim=1)
                    v1_reconstructed, v2_reconstructed, scalars_reconstructed = unetdecoder(arrays_con)
                    
                    # Second encoder pass
                    outputs_2 = encoderunet(v1_reconstructed.unsqueeze(1),
                                          v2_reconstructed.unsqueeze(1),
                                          scalars_reconstructed.unsqueeze(1))
                    
                    # Compute "second pass forward loss"
                    l1_loss_per_element = F.l1_loss(outputs_2, arrays, reduction='none')
                    l1_loss_per_sample = l1_loss_per_element.sum(dim=[2, 3]).squeeze()
                    weight = 1/scalars_reconstructed.unsqueeze(1)[:,0,0]
                    loss_for_2 = (l1_loss_per_sample * weight).mean()
                    
                    # Second decoder pass setup
                    arrays_p = outputs[:-1]
                    main_batch = outputs[1:]
                    arrays_con_2 = torch.cat([arrays_p, main_batch], dim=1)
                    v1_reconstructed_2, v2_reconstructed_2, scalars_reconstructed_2 = unetdecoder(arrays_con_2)
                    
                    # Prepare tensors
                    v1, v2 = v1.squeeze(1), v2.squeeze(1)
                    v1_weight, v2_weight = v1_weight.squeeze(1), v2_weight.squeeze(1)
                    scalars = scalars.squeeze(1)
                    
                    # Penalty losses
                    penalty_loss = torch.where(v2_reconstructed < v1_reconstructed,
                                             v1_reconstructed - v2_reconstructed,
                                             torch.zeros_like(v2_reconstructed)).sum()
                    
                    penalty_loss_2 = torch.where(v2_reconstructed_2 < v1_reconstructed_2,
                                               v1_reconstructed_2 - v2_reconstructed_2,
                                               torch.zeros_like(v2_reconstructed_2)).sum()
                    
                    # Consistency losses
                    if v1_reconstructed.size(0) > 1:
                        mse_loss_v1_diff = weighted_l1_loss(
                            v1_reconstructed[:-1, -52:],
                            v1_reconstructed[1:, :52],
                            v1_weight[:-1, -52:]
                        )
                        mse_loss_v2_diff = weighted_l1_loss(
                            v2_reconstructed[:-1, -52:],
                            v2_reconstructed[1:, :52],
                            v2_weight[:-1, -52:]
                        )
                        mse_loss_v1_diff_2 = weighted_l1_loss(
                            v1_reconstructed_2[:-1, -52:],
                            v1_reconstructed_2[1:, :52],
                            v1_weight[1:-1, -52:]
                        )
                        mse_loss_v2_diff_2 = weighted_l1_loss(
                            v2_reconstructed_2[:-1, -52:],
                            v2_reconstructed_2[1:, :52],
                            v2_weight[1:-1, -52:]
                        )
                        
                        consistency_loss = (mse_loss_v1_diff + mse_loss_v2_diff +
                                          mse_loss_v1_diff_2 + mse_loss_v2_diff_2)
                    
                    # Reconstruction losses
                    mse_loss_v1 = weighted_l1_loss(v1_reconstructed, v1, v1_weight)
                    mse_loss_v2 = weighted_l1_loss(v2_reconstructed, v2, v2_weight)
                    mse_loss_scalars = criterion(scalars_reconstructed, scalars) * 5
                    
                    loss_back = (mse_loss_v1 + mse_loss_v2 + mse_loss_scalars +
                                penalty_loss * 10)
                    
                    mse_loss_v1_2 = weighted_l1_loss(v1_reconstructed_2, v1[1:], v1_weight[1:])
                    mse_loss_v2_2 = weighted_l1_loss(v2_reconstructed_2, v2[1:], v2_weight[1:])
                    mse_loss_scalars_2 = criterion(scalars_reconstructed_2, scalars[1:]) * 5
                    
                    loss_back_2 = (mse_loss_v1_2 + mse_loss_v2_2 + mse_loss_scalars_2 +
                                  penalty_loss_2 * 10)
                    
                    # Total loss
                    loss = loss_for + loss_back + loss_for_2 + loss_back_2 + consistency_loss
                    
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
            
            torch.save(checkpoint, 'Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_embedded_64_Attention_vcap_checkpoint.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies           

    #######################################################################




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    dist.init_process_group(
        "nccl", 
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
        batch_size = 256// world_size  # Scale batch size by number of GPUs
    

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
    world_size = 2
    mp.spawn(
        train_ddp,
        args=(world_size, generate_flag, KM),
        nprocs=world_size,
        join=True
    )