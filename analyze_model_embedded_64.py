import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
from generate_and_train_amp_parallel_coll0_embedded_64_local import (
    EncoderUNet, UNetDecoder, CustomDataset, load_dataset
)
import os
import scipy.io
from torch.utils.data import DataLoader
from collections import defaultdict

def load_trained_models(checkpoint_path, device):
    """Load trained models from checkpoint."""
    # Initialize models
    vector_dim = 104
    scalar_count = 5
    latent_image_size = 128
    
    # Initialize encoder
    encoderunet = EncoderUNet(vector_dim, scalar_count)
    
    # Initialize decoder
    unetdecoder = UNetDecoder(vector_dim, scalar_count)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dictionaries
    encoderunet.load_state_dict(checkpoint['encoderunet_state_dict'])
    unetdecoder.load_state_dict(checkpoint['unetdecoder_state_dict'])
    
    return encoderunet.to(device), unetdecoder.to(device)

def analyze_parameters(model, model_name):
    """Analyze model parameters distribution."""
    stats = defaultdict(dict)
    has_nans = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check for NaNs
            if torch.isnan(param.data).any():
                print(f"WARNING: Found NaNs in parameter {name}")
                has_nans = True
            
            # For parameters with more than 1 value
            if param.numel() > 1:
                stats[name]['mean'] = param.data.mean().item()
                stats[name]['std'] = param.data.std().item()
                stats[name]['min'] = param.data.min().item()
                stats[name]['max'] = param.data.max().item()
            else:
                # Special handling for single-value parameters
                stats[name]['value'] = param.data.item()
            
            # Plot histogram
            plt.figure(figsize=(10, 4))
            plt.hist(param.data.cpu().numpy().flatten(), bins=50)
            plt.title(f'{model_name} - {name} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.savefig(f'analysis_results/{model_name}/parameters/{name.replace(".", "_")}_dist.png')
            plt.close()
    
    if not has_nans:
        print(f"No NaNs found in {model_name} parameters")
    return stats

def analyze_activations(model, input_batch, model_name):
    """Analyze activations throughout the model."""
    activations = {}
    has_nans = False
    hooks = []  # Initialize hooks list
    
    def hook_fn(name):
        def hook(module, input, output):
            if torch.isnan(output).any():
                print(f"WARNING: Found NaNs in activation {name}")
                nonlocal has_nans  # Use nonlocal instead of global
                has_nans = True
            activations[name] = output.detach()
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(*input_batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze activations
    activation_stats = {}
    for name, activation in activations.items():
        if activation.dim() > 1:  # Skip 1D tensors
            activation_stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item() if activation.numel() > 1 else 'N/A (single value)',
                'sparsity': (activation == 0).float().mean().item(),
                'shape': list(activation.shape)
            }
            
            # Plot activation distribution
            plt.figure(figsize=(10, 4))
            plt.hist(activation.cpu().numpy().flatten(), bins=50)
            plt.title(f'{model_name} - {name} Activation Distribution')
            plt.xlabel('Activation Value')
            plt.ylabel('Count')
            plt.savefig(f'analysis_results/{model_name}/activations/{name.replace(".", "_")}_dist.png')
            plt.close()
            
            # If the activation is from a convolutional layer, visualize feature maps
            if activation.dim() == 4:
                # Take first batch and up to 16 channels
                feature_maps = activation[0, :min(16, activation.size(1))]
                grid = make_grid(feature_maps.unsqueeze(1), normalize=True, nrow=4)
                plt.figure(figsize=(12, 12))
                plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
                plt.title(f'{name} Feature Maps')
                plt.axis('off')
                plt.savefig(f'analysis_results/{model_name}/feature_maps/{name.replace(".", "_")}_maps.png')
                plt.close()
    
    if not has_nans:
        print(f"No NaNs found in {model_name} activations")
    return activation_stats

def generate_analysis_report(param_stats, activation_stats, model_name):
    """Generate a text report of the analysis."""
    with open(f'analysis_results/{model_name}/analysis_report_ResBlock.txt', 'w') as f:
        f.write(f"Analysis Report for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Parameter Statistics:\n")
        f.write("-" * 20 + "\n")
        for param_name, stats in param_stats.items():
            f.write(f"\n{param_name}:\n")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value:.6f}\n")
        
        f.write("\nActivation Statistics:\n")
        f.write("-" * 20 + "\n")
        for layer_name, stats in activation_stats.items():
            f.write(f"\n{layer_name}:\n")
            for stat_name, value in stats.items():
                if stat_name == 'shape':
                    f.write(f"  {stat_name}: {value}\n")
                else:
                    f.write(f"  {stat_name}: {value:.6f}\n")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_embedded_64_impLoss_checkpoint.pth'
    
    # Create directories for results
    for model_name in ['encoder', 'decoder']:
        for subdir in ['parameters', 'activations', 'feature_maps']:
            os.makedirs(f'analysis_results/{model_name}/{subdir}', exist_ok=True)
    
    # Load models
    encoderunet, unetdecoder = load_trained_models(checkpoint_path, device)
    encoderunet.eval()
    unetdecoder.eval()
    
    # Load one dataset for testing
    dataset = load_dataset(0)  # Load first dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    batch = [b.to(device) for b in batch]
    
    # Analyze encoder
    print("Analyzing encoder...")
    encoder_param_stats = analyze_parameters(encoderunet, 'encoder')
    encoder_activation_stats = analyze_activations(encoderunet, batch[:3], 'encoder')
    generate_analysis_report(encoder_param_stats, encoder_activation_stats, 'encoder')
    
    # Analyze decoder
    print("Analyzing decoder...")
    decoder_param_stats = analyze_parameters(unetdecoder, 'decoder')
    # For decoder, we need to prepare the input
    with torch.no_grad():
        encoder_output = encoderunet(*batch[:3])
        arrays_p = encoder_output[:-1]
        main_batch = encoder_output[1:]
        arrays_con = torch.cat([arrays_p, main_batch], dim=1)
    decoder_activation_stats = analyze_activations(unetdecoder, (arrays_con,), 'decoder')
    generate_analysis_report(decoder_param_stats, decoder_activation_stats, 'decoder')
    
    print("Analysis complete! Results saved in analysis_results/")

if __name__ == "__main__":
    main() 