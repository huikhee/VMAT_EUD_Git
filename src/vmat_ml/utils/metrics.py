import torch
import torch.nn.functional as F
import numpy as np

def create_gaussian_kernel(size, sigma, device):
    """Create a 2D Gaussian kernel.
    
    Args:
        size (int): Size of the kernel
        sigma (float): Standard deviation of the Gaussian
        device: PyTorch device
        
    Returns:
        torch.Tensor: Gaussian kernel
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
    """Calculate the 2D gamma index using PyTorch tensors.
    
    Args:
        ref_data (torch.Tensor): Reference dose distribution
        eval_data (torch.Tensor): Evaluated dose distribution
        dose_threshold (float): Dose difference threshold as fraction
        distance_mm (float): Distance-to-agreement threshold in mm
        pixel_spacing (tuple): Pixel spacing in mm
        
    Returns:
        float: Gamma passing rate as percentage
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
        dose_diff = dose_diff.unsqueeze(0)

    # Gaussian smoothing for distance-to-agreement
    kernel_size = int(distance_mm / min(pixel_spacing) * 2 + 1)
    sigma = distance_mm / min(pixel_spacing) / 2
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma, ref_data.device)

    # Apply convolution with 'same' padding
    distance_agreement = F.conv2d(dose_diff, gaussian_kernel, padding='same')

    # Calculate gamma index
    gamma_index = torch.sqrt((dose_diff / dose_threshold)**2 + (distance_agreement / distance_mm)**2)

    # Calculate gamma passing rate
    gamma_passing_rate = (gamma_index < 1).float().mean().item() * 100

    return gamma_passing_rate

def calculate_gamma_index_accurate(ref_data, eval_data, dose_threshold=0.03, distance_mm=3, pixel_spacing=(2.5, 2.5)):
    """Calculate the gamma index without convolution and without normalizing to maximum dose.
    
    Args:
        ref_data (torch.Tensor): Reference dose distribution
        eval_data (torch.Tensor): Evaluated dose distribution
        dose_threshold (float): Dose difference threshold as fraction
        distance_mm (float): Distance-to-agreement threshold in mm
        pixel_spacing (tuple): Pixel spacing in mm
        
    Returns:
        float: Gamma passing rate as percentage
    """
    assert ref_data.shape == eval_data.shape, "Reference and evaluated data must have the same shape"

    # Prepare for gamma index calculation
    distance_threshold = distance_mm / np.array(pixel_spacing)
    dose_diff_threshold = dose_threshold

    gamma_map = torch.full(eval_data.shape, float('inf'))

    # Iterate over each pixel in the evaluated data
    for i in range(eval_data.shape[0]):
        for j in range(eval_data.shape[1]):
            # Compute dose difference and distance for all points in the reference data
            dose_diff = torch.abs(ref_data - eval_data[i, j])
            y_distance, x_distance = torch.meshgrid(torch.arange(ref_data.shape[0]), 
                                                  torch.arange(ref_data.shape[1]), 
                                                  indexing='ij')
            y_distance = (y_distance - i) * pixel_spacing[0]
            x_distance = (x_distance - j) * pixel_spacing[1]
            distance = torch.sqrt(torch.pow(y_distance, 2) + torch.pow(x_distance, 2))

            # Compute gamma index for each point
            gamma_index = torch.sqrt(torch.pow(dose_diff / dose_diff_threshold, 2) + 
                                   torch.pow(distance / distance_threshold, 2))

            # Find the minimum gamma index for the current evaluated point
            gamma_map[i, j] = torch.min(gamma_index)

    # Compute the passing rate
    passing_criteria = (gamma_map <= 1).float()
    gamma_passing_rate = torch.mean(passing_criteria).item() * 100

    return gamma_passing_rate 