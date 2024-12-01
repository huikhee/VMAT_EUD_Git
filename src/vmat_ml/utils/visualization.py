import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pymedphys

def create_gamma_colormap():
    """Create a colormap for gamma index visualization.
    
    Returns:
        tuple: (colormap, norm) for matplotlib plotting
    """
    colors = ['green', 'red']
    cmap = mcolors.ListedColormap(colors)
    boundaries = [0, 1, 2]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm

def visualize_results(sample_idx, dataset, encoder_unet, unet_decoder, device, row_index=35):
    """Visualize the results of the model predictions.
    
    Args:
        sample_idx (int): Index of the sample to visualize
        dataset: Dataset containing the samples
        encoder_unet: Encoder-UNet model
        unet_decoder: UNet-Decoder model
        device: PyTorch device
        row_index (int): Row index for profile visualization
    """
    v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p = dataset[sample_idx]
    v1, v2, scalars, arrays, arrays_p = v1.to(device), v2.to(device), scalars.to(device), arrays.to(device), arrays_p.to(device)
    arrays = arrays.squeeze(1)
    arrays_p = arrays_p.squeeze(1)

    # Forward pass
    encoder_unet.eval()
    with torch.no_grad():
        outputs = encoder_unet(v1.unsqueeze(0), v2.unsqueeze(0), scalars.unsqueeze(0))

    # Calculate gamma index
    gamma_options = {
        'dose_percent_threshold': 3,
        'distance_mm_threshold': 3,
        'lower_percent_dose_cutoff': 20,
        'interp_fraction': 10,
        'max_gamma': 2,
        'random_subset': None,
        'local_gamma': True,
        'ram_available': 2**29,
        'quiet': True
    }

    grid = 2
    xmin, xmax = -130, 130
    ymin, ymax = -130, 130
    x = np.arange(xmin, xmax + grid, grid)
    y = np.arange(ymin, ymax + grid, grid)
    coords = (y, x)

    gamma = pymedphys.gamma(
        coords, arrays.squeeze().cpu().numpy(),
        coords, outputs.squeeze().cpu().numpy(),
        **gamma_options)

    valid_gamma = gamma[~np.isnan(gamma)]
    gamma_passing = 100 * np.round(np.sum(valid_gamma <= 1) / len(valid_gamma), 4)

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    # Plot actual frame
    axes[0].imshow(arrays.squeeze().cpu().numpy())
    axes[0].set_title('Actual Frame')

    # Plot predicted frame
    axes[1].imshow(outputs.squeeze().cpu().numpy())
    axes[1].set_title('Predicted Frame')

    # Plot gamma index
    cmap, norm = create_gamma_colormap()
    axes[2].imshow(gamma, cmap=cmap, norm=norm)
    axes[2].set_title(f'Passing rate = {round(gamma_passing, 2)}%')

    # Plot horizontal line profile
    line_profile_arrays = arrays.squeeze().cpu().numpy()[row_index,:]
    line_profile_outputs = outputs.squeeze().cpu().numpy()[row_index,:]
    axes[3].plot(line_profile_arrays, color='blue', label='Actual')
    axes[3].plot(line_profile_outputs, color='red', label='Predicted')
    axes[3].set_title('Horizontal Line Profile')
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    # Backward pass visualization
    arrays_con = torch.cat([arrays_p.unsqueeze(0), arrays.unsqueeze(0)], dim=1)
    with torch.no_grad():
        v1_reconstructed, v2_reconstructed, scalars_reconstructed = unet_decoder(arrays_con)

    # Create second visualization for MLC positions
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

    # Plot MLC positions
    x_values_extended = np.linspace(-130, 130, 53)
    
    # Previous positions
    v1_values = v1.squeeze(0).cpu().numpy()[:52]
    v2_values = v2.squeeze(0).cpu().numpy()[:52]
    v1_reconstructed_values = v1_reconstructed.cpu().numpy()[:52]
    v2_reconstructed_values = v2_reconstructed.cpu().numpy()[:52]

    axes[0].step(v1_values, -x_values_extended, where='post', label='Original', color='blue')
    axes[0].step(v1_reconstructed_values, -x_values_extended, where='post', label='Reconstructed', color='red', linestyle='--')
    axes[0].step(v2_values, -x_values_extended, where='post', color='blue')
    axes[0].step(v2_reconstructed_values, -x_values_extended, where='post', color='red', linestyle='--')
    axes[0].set_title('Previous MLC Positions')
    axes[0].legend()

    # Current positions
    v1_values = v1.squeeze(0).cpu().numpy()[52:]
    v2_values = v2.squeeze(0).cpu().numpy()[52:]
    v1_reconstructed_values = v1_reconstructed.cpu().numpy()[52:]
    v2_reconstructed_values = v2_reconstructed.cpu().numpy()[52:]

    axes[1].step(v1_values, -x_values_extended, where='post', label='Original', color='blue')
    axes[1].step(v1_reconstructed_values, -x_values_extended, where='post', label='Reconstructed', color='red', linestyle='--')
    axes[1].step(v2_values, -x_values_extended, where='post', color='blue')
    axes[1].step(v2_reconstructed_values, -x_values_extended, where='post', color='red', linestyle='--')
    axes[1].set_title('Current MLC Positions')
    axes[1].legend()

    # Plot weights
    axes[2].step(v1_weight.squeeze(0).cpu().numpy()[52:], -x_values_extended, where='post', color='blue')
    axes[2].step(v2_weight.squeeze(0).cpu().numpy()[52:], -x_values_extended, where='post', color='blue')
    axes[2].set_title('MLC Weights')

    # Plot MU comparison
    bar_width = 0.3
    index = np.arange(1)
    space = 0.05
    axes[3].bar(index, scalars[0, 0].cpu().numpy(), bar_width, label='Original MU')
    axes[3].bar(index + bar_width + space, scalars_reconstructed[0].cpu().numpy(), bar_width, label='Reconstructed MU')
    axes[3].set_title('Monitor Units')
    axes[3].legend()

    plt.tight_layout()
    plt.show() 