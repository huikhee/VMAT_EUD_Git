# Combined u-net and decoder architecture
# cross training
# same as used for furst paper submission
#now adapted for VMAT
import sys


import numpy as np
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F 


from torch.utils.data import DataLoader,Dataset,random_split,Subset
#from torchvision import transforms
#from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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

from torch.cuda.amp import autocast, GradScaler

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
    
    num_samples = 2000
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
    
    num_samples = 2000
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
    
    num_samples = 2000
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
        rotated_matrix = scipy.ndimage.rotate(matrix, 45, reshape=False, mode='constant', cval=0.0)
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
    filename = os.path.join("VMAT_Art_data", f"Art_dataset_coll45_{dataset_num}.pt")
    torch.save(dataset, filename)

def load_dataset(dataset_num):
    """Function to load dataset"""
    filename = os.path.join("VMAT_Art_data", f"Art_dataset_coll45_{dataset_num}.pt")
    if os.path.exists(filename):
        try:
            return torch.load(filename)
        except Exception as e:
            print(f"Error loading dataset {dataset_num}: {str(e)}")
            return None
    return None

def generate_and_save_dataset(dataset_num):
    """Generate and save a complete dataset."""
    # Choose the appropriate vector generation function based on dataset number
    if 0 <= dataset_num <= 79:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = \
            generate_random_vectors_scalar_regular(42 + dataset_num)
    elif 80 <= dataset_num <= 159:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = \
            generate_random_vectors_scalar_semiregular(42 + dataset_num)
    else:
        vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = \
            generate_random_vectors_scalars(42 + dataset_num)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtEncoder(nn.Module):
    def __init__(self, vector_dim, scalar_count, latent_image_size):
        super(ExtEncoder, self).__init__()
        self.vector_fc = nn.Linear(vector_dim * 2, 512)
        self.scalar_fc = nn.ModuleList([nn.Linear(1, 64) for _ in range(scalar_count)])
        self.combined_fc = nn.Linear(512 + scalar_count * 64, latent_image_size ** 2 * 1)
        self.latent_image_size = latent_image_size
        
    def forward(self, vector1, vector2, scalars):
        # Process vectors
        vectors_combined = torch.cat((vector1.flatten(1), vector2.flatten(1)), dim=1)
        vectors_encoded = F.relu(self.vector_fc(vectors_combined))
        
        # Process scalars
        scalars = scalars.squeeze(1)  # Remove the extra dimension
        scalars_encoded = []
        for i, fc in enumerate(self.scalar_fc):
            scalar = scalars[:, i].unsqueeze(1)  # Get i-th scalar and add dimension for Linear layer
            scalar_encoded = F.relu(fc(scalar))
            scalars_encoded.append(scalar_encoded)
        
        # Combine all encoded features
        scalars_encoded = torch.cat(scalars_encoded, dim=1)
        combined = torch.cat((vectors_encoded, scalars_encoded), dim=1)
        
        # Final processing
        latent_image = F.relu(self.combined_fc(combined))
        return latent_image.view(-1, 1, self.latent_image_size, self.latent_image_size)

class ExtDecoder(nn.Module):
    def __init__(self, vector_dim, scalar_count, latent_image_size):
        super(ExtDecoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)
        self.fc1 = nn.Linear(latent_image_size ** 2, 512)
        self.vector_fc = nn.Linear(512, vector_dim * 2)
        self.scalar_fc = nn.Linear(512, scalar_count)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        vectors = self.vector_fc(x)
        scalars = self.scalar_fc(x)
        vector1, vector2 = torch.split(vectors, vectors.size(1)//2, dim=1)
        return vector1, vector2, scalars

class EncoderUNet(nn.Module):
    def __init__(self, ExtEncoder, vector_dim, scalar_count, latent_image_size, 
                 in_channels, out_channels, resize_out, freeze_encoder=False):
        super(EncoderUNet, self).__init__()
        self.encoder = ExtEncoder(vector_dim, scalar_count, latent_image_size)
        self.resize_out = resize_out
        
        # U-Net components
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(self, vector1, vector2, scalars):
        x = self.encoder(vector1, vector2, scalars)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.resize_out != logits.size(-1):
            logits = F.interpolate(logits, size=(self.resize_out, self.resize_out), 
                                 mode='bilinear', align_corners=True)
        return logits

class UNetDecoder(nn.Module):
    def __init__(self, ExtDecoder, vector_dim, scalar_count, latent_image_size, 
                 in_channels, out_channels, resize_in, freeze_encoder=False):
        super(UNetDecoder, self).__init__()
        self.decoder = ExtDecoder(vector_dim, scalar_count, latent_image_size)
        self.resize_in = resize_in
        
        # U-Net components
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)
        
        if freeze_encoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        if x.size(-1) != self.resize_in:
            x = F.interpolate(x, size=(self.resize_in, self.resize_in), 
                            mode='bilinear', align_corners=True)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        vector1, vector2, scalars = self.decoder(logits)
        return vector1, vector2, scalars

# U-Net helper components
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
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

def weighted_mse_loss(input, target, weights):
    squared_error = (input - target) ** 2
    weighted_squared_error = squared_error * weights
    loss = weighted_squared_error.mean()
    return loss

def weighted_l1_loss(input, target, weights):
    absolute_error = torch.abs(input - target)
    weighted_absolute_error = absolute_error * weights
    return weighted_absolute_error.mean()

def initialize_weights(model):
    """Initialize the weights of the model using Xavier initialization"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def setup_training(encoderunet, unetdecoder, resume=0):
    lr = 1e-4
    criterion = nn.MSELoss().to(device)
    
    if resume == 1:
        checkpoint = torch.load('Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_coll45_checkpoint.pth', 
                              map_location=device)
        encoderunet.load_state_dict(checkpoint['encoderunet_state_dict'])
        unetdecoder.load_state_dict(checkpoint['unetdecoder_state_dict'])
        
        optimizer = AdamW(list(encoderunet.parameters()) + list(unetdecoder.parameters()), 
                         lr=lr, weight_decay=1e-2)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if it exists
        scaler = GradScaler()
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        initialize_weights(unetdecoder)
        initialize_weights(encoderunet)
        
        start_epoch = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        optimizer = AdamW(list(encoderunet.parameters()) + list(unetdecoder.parameters()), 
                         lr=lr, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)
    
    return (optimizer, scheduler, criterion, start_epoch, train_losses, val_losses, 
            train_accuracies, val_accuracies)


# training_loop.py

def train_cross(encoderunet, unetdecoder, train_loaders, val_loaders, device, resume=0):
    # Setup training parameters
    EPOCHS = 300
    optimizer, scheduler, criterion, start_epoch, train_losses, val_losses, train_accuracies, val_accuracies = setup_training(encoderunet, unetdecoder, resume)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Print settings
    print('SETTINGS')
    print('epochs:', EPOCHS)
    print('batch size:', batch_size)
    print('optimizer: Adam')
    print('learning rate:', optimizer.param_groups[0]['lr'])
    print('loss: weighted L1 + MSE')
    print('mixed precision: enabled')

    sys.stdout.flush()
    
    line_length = 155
    
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        
        # Training
        encoderunet.train()
        unetdecoder.train()
        
        running_train_losses = [0.0] * len(train_loaders)
        running_train_accuracies = [0.0] * len(train_loaders)
        
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
                
                # Use autocast for mixed precision training
                with autocast():
                    # Forward pass through encoder-unet
                    outputs = encoderunet(v1, v2, scalars)
                    accuracy = calculate_gamma_index(outputs, arrays)
                    
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
                
                loader_loss_sum += loss.item()
                loader_accuracy_sum += accuracy
            
            # Calculate averages for this loader
            num_batches = len(train_loader)
            running_train_losses[i] = loader_loss_sum / num_batches
            running_train_accuracies[i] = loader_accuracy_sum / num_batches


        # Validation
        encoderunet.eval()
        unetdecoder.eval()
        
        with torch.no_grad():
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
                    accuracy = calculate_gamma_index(outputs, arrays)
                    
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
                    
                    #print(f"Epoch [{epoch+1}/{EPOCHS}] Loader [{i+1}/{len(val_loaders)}] "
                    #      f"Batch [{batch_idx+1}/{len(val_loader)}]  "
                    #      f"Temp. Val. Loss: {loss_for.item():.2e} {loss_back.item():.2e} "
                    #      f"{loss_for_2.item():.2e} {loss_back_2.item():.2e} {consistency_loss.item():.2e} "
                    #      f"{loss.item():.2e}  Temp. Val. Acc.: {accuracy:.2f}".ljust(line_length), end='\r')
                    
                    loader_loss_sum += loss.item()
                    loader_accuracy_sum += accuracy
                
                # Calculate averages for this loader
                num_batches = len(val_loader)
                running_val_losses[i] = loader_loss_sum / num_batches
                running_val_accuracies[i] = loader_accuracy_sum / num_batches


             # Calculate average metrics for the epoch
        average_train_loss = sum(running_train_losses) / len(train_loaders)
        average_train_accuracy = sum(running_train_accuracies) / len(train_loaders)
        average_val_loss = sum(running_val_losses) / len(val_loaders)
        average_val_accuracy = sum(running_val_accuracies) / len(val_loaders)

        # Append to history
        train_losses.append(average_train_loss)
        train_accuracies.append(average_train_accuracy)
        val_losses.append(average_val_loss)
        val_accuracies.append(average_val_accuracy)
        
        # Step scheduler
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Avg. Train Loss: {average_train_loss:.4e} "
              f"Avg. Train Accuracy: {average_train_accuracy:.2f} "
              f"Avg. Val. Loss: {average_val_loss:.4e} "
              f"Avg. Val. Accuracy: {average_val_accuracy:.2f} "
              f"Elap. Time: {elapsed_time:.1f} seconds "
              f"Current LR: {current_lr:.4e}")
        
        sys.stdout.flush()

        # Save checkpoint with scaler state
        checkpoint = {
            'epoch': epoch,
            'encoderunet_state_dict': encoderunet.state_dict(),
            'unetdecoder_state_dict': unetdecoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Save scaler state
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
        }
        
        torch.save(checkpoint, 'Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_coll45_checkpoint.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies           

    #######################################################################
if __name__ == "__main__":
    # Load KM matrix
    KM_data = scipy.io.loadmat('data/KM_1500.mat')
    KM = KM_data['KM_1500']

    # Set generation flag
    generate_flag = 0  # Set this flag to 1 if you want to generate datasets again

    # Create directories if they don't exist
    os.makedirs("VMAT_Art_data", exist_ok=True)
    os.makedirs("Cross_CP", exist_ok=True)

    train_loaders = []
    val_loaders = []

    # Generate or load datasets
    for dataset_num in range(0, 18, 1):
        start_time = time.time()

        if generate_flag == 0:
            Art_dataset = load_dataset(dataset_num)
            if Art_dataset is None:
                Art_dataset = generate_and_save_dataset(dataset_num)
                print(f"Generated and saved dataset {dataset_num} because it was not found on disk")
            else:
                print(f"Loaded dataset {dataset_num} from disk")
        else:
            Art_dataset = generate_and_save_dataset(dataset_num)
            print(f"Generated and saved dataset {dataset_num}")

        sys.stdout.flush()
        
        # Split into train and validation sets
        VALIDSPLIT = 0.8
        dataset_size = len(Art_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(VALIDSPLIT * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]

        train_ds = Subset(Art_dataset, train_indices)
        val_ds = Subset(Art_dataset, val_indices)

        batch_size = 128
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        end_time = time.time()
        print(f"Dataset {dataset_num} processing time: {end_time - start_time:.2f} seconds")
        print(f"Dataset {dataset_num} is done")



    vector_dim = 104
    scalar_count = 5
    latent_image_size = 128
    in_channels = 1
    out_channels = 1
    resize_out = 131

    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    encoderunet = EncoderUNet(ExtEncoder, vector_dim, scalar_count, latent_image_size, 
                             in_channels, out_channels, resize_out, freeze_encoder=False)
    encoderunet = encoderunet.to(device)  # Explicitly move to MPS

    vector_dim = 104
    scalar_count = 5
    latent_image_size = 128
    in_channels = 2
    out_channels = 1
    resize_in = 128

    unetdecoder = UNetDecoder(ExtDecoder, vector_dim, scalar_count, latent_image_size,
                             in_channels, out_channels, resize_in, freeze_encoder=False)
    unetdecoder = unetdecoder.to(device)  # Explicitly move to MPS

    print("\nModel devices:")
    print(f"encoderunet device: {next(encoderunet.parameters()).device}")
    print(f"unetdecoder device: {next(unetdecoder.parameters()).device}")
    # Start training
    print("Starting training...")
    train_cross(encoderunet, unetdecoder, train_loaders, val_loaders, device, resume=0)
    print("Training completed!")