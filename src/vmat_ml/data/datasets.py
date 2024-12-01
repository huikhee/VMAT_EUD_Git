import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.ndimage

def interpolate_vectors(v1_start, v1_end, v2_start, v2_end, s2_start, s2_end, s3_start, s3_end, num_interpolations=0):
    """Interpolate between two sets of vectors and scalars.
    
    Args:
        v1_start, v1_end: Start and end points for vector1
        v2_start, v2_end: Start and end points for vector2
        s2_start, s2_end: Start and end points for scalar2
        s3_start, s3_end: Start and end points for scalar3
        num_interpolations: Number of interpolation points
        
    Returns:
        tuple: (interpolated_v1, interpolated_v2, interpolated_s2, interpolated_s3)
    """
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

def create_boundary_matrix(vector1, vector2, scalar1, scalar2, scalar3):
    """Create boundary matrix from vectors and scalars.
    
    Args:
        vector1, vector2: MLC leaf position vectors
        scalar1, scalar2, scalar3: Scalar values
        
    Returns:
        list: Collection of matrices
    """
    vector1_int = np.round(vector1).astype(int)
    vector2_int = np.round(vector2).astype(int)
    scalar2_int = np.round(scalar2).astype(int)
    scalar3_int = np.round(scalar3).astype(int)

    num_samples = len(scalar2_int)
    matrix_collection = []
    
    for i in range(num_samples):
        matrix = np.zeros((261, 261))

        for bin_index in range(52):
            y_start = max(-130 + bin_index * 5, -130)
            y_end = min(y_start + 5, 130)
            matrix[y_start+130:y_end+130, vector1_int[i,bin_index]+130:vector2_int[i,bin_index]+130] = 1

        matrix[:max(-130, int(scalar2_int[i])) + 130, :] = 0
        matrix[min(130, int(scalar3_int[i])) + 130:, :] = 0
        
        matrix = np.flipud(matrix)
        rotated_matrix = scipy.ndimage.rotate(matrix, 45, reshape=False, mode='constant', cval=0.0)
        matrix_collection.append(rotated_matrix)

    return matrix_collection

class CustomDataset(Dataset):
    """Dataset for artificial VMAT data."""
    
    def __init__(self, vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight, arrays):
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

        v1 = torch.cat([self.vector1[prev_idx].unsqueeze(0), self.vector1[idx].unsqueeze(0)], dim=1)
        v2 = torch.cat([self.vector2[prev_idx].unsqueeze(0), self.vector2[idx].unsqueeze(0)], dim=1)
        v1_weight = torch.cat([self.vector1_weight[prev_idx].unsqueeze(0), self.vector1_weight[idx].unsqueeze(0)], dim=1)
        v2_weight = torch.cat([self.vector2_weight[prev_idx].unsqueeze(0), self.vector2_weight[idx].unsqueeze(0)], dim=1)

        scalar1 = self.scalar1[idx].unsqueeze(0).unsqueeze(0)
        scalar2_current = self.scalar2[idx].unsqueeze(0).unsqueeze(0)
        scalar2_previous = self.scalar2[prev_idx].unsqueeze(0).unsqueeze(0)
        scalar3_current = self.scalar3[idx].unsqueeze(0).unsqueeze(0)
        scalar3_previous = self.scalar3[prev_idx].unsqueeze(0).unsqueeze(0)

        scalars = torch.cat([scalar1, scalar2_previous, scalar2_current, scalar3_previous, scalar3_current], dim=1)

        arrays = self.arrays[idx].unsqueeze(0)
        arrays_p = self.arrays[prev_idx].unsqueeze(0)

        return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p

class RealVMATDataset(Dataset):
    """Dataset for real VMAT data."""
    
    def __init__(self, segmentMU, mlcx1, mlcx2, y1, y2, accumulated_frames):
        self.segmentMU = torch.tensor(segmentMU, dtype=torch.float32)
        self.mlcx1 = torch.tensor(mlcx1, dtype=torch.float32)
        self.mlcx2 = torch.tensor(mlcx2, dtype=torch.float32)
        self.y1 = torch.tensor(y1, dtype=torch.float32)
        self.y2 = torch.tensor(y2, dtype=torch.float32)
        self.arrays = 125 * torch.tensor(accumulated_frames, dtype=torch.float32).unsqueeze(1)
        
        # Extend by adding a copy of the first element at the beginning
        first_element = self.arrays[0].unsqueeze(0)
        self.arrays = torch.cat((first_element, self.arrays), dim=0)
        self.arrays = torch.flip(self.arrays, dims=[2])

    def __len__(self):
        return self.mlcx1.size(0) - 2

    def __getitem__(self, idx):
        idx += 2
        prev_idx = idx - 1

        v1 = torch.cat([self.mlcx1[prev_idx].unsqueeze(0), self.mlcx1[idx].unsqueeze(0)], dim=1)
        v2 = torch.cat([self.mlcx2[prev_idx].unsqueeze(0), self.mlcx2[idx].unsqueeze(0)], dim=1)

        segmentMU = self.segmentMU[idx].unsqueeze(0).unsqueeze(0)
        y1_prev = self.y1[prev_idx].unsqueeze(0).unsqueeze(0)
        y1_cur = self.y1[idx].unsqueeze(0).unsqueeze(0)
        y2_prev = self.y2[prev_idx].unsqueeze(0).unsqueeze(0)
        y2_cur = self.y2[idx].unsqueeze(0).unsqueeze(0)

        # Initialize weighting vectors
        vector1_weight_prev = torch.ones((52), dtype=torch.float32) * 0
        vector2_weight_prev = torch.ones((52), dtype=torch.float32) * 0
        vector1_weight_cur = torch.ones((52), dtype=torch.float32) * 0
        vector2_weight_cur = torch.ones((52), dtype=torch.float32) * 0

        # Calculate weights for previous vectors
        lower_limit = int(np.ceil((130 + self.y1[prev_idx].item()) / 5))
        upper_limit = int(np.ceil((130 + self.y2[prev_idx].item()) / 5))
        lower_limit_weight = max(0, lower_limit-2)
        upper_limit_weight = min(52, upper_limit+2)
        vector1_weight_prev[lower_limit_weight:upper_limit_weight+1] = 1.0
        vector2_weight_prev[lower_limit_weight:upper_limit_weight+1] = 1.0

        # Calculate weights for current vectors
        lower_limit = int(np.ceil((130 + self.y1[idx].item()) / 5))
        upper_limit = int(np.ceil((130 + self.y2[idx].item()) / 5))
        lower_limit_weight = max(0, lower_limit-2)
        upper_limit_weight = min(52, upper_limit+2)
        vector1_weight_cur[lower_limit_weight:upper_limit_weight+1] = 1.0
        vector2_weight_cur[lower_limit_weight:upper_limit_weight+1] = 1.0

        v1_weight = torch.cat([vector1_weight_prev.unsqueeze(0), vector1_weight_cur.unsqueeze(0)], dim=1)
        v2_weight = torch.cat([vector2_weight_prev.unsqueeze(0), vector2_weight_cur.unsqueeze(0)], dim=1)

        scalars = torch.cat([segmentMU, y1_prev, y1_cur, y2_prev, y2_cur], dim=1)

        arrays = self.arrays[idx].unsqueeze(0)
        arrays_p = self.arrays[prev_idx].unsqueeze(0)

        return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p 