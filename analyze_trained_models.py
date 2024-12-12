import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from generate_and_train_amp_parallel_coll0 import (
    EncoderUNet, UNetDecoder, ExtEncoder, ExtDecoder,
    CustomDataset, load_dataset
)
import torch.nn.functional as F

def save_to_file(content, filepath):
    """Helper function to save content to a file."""
    with open(filepath, 'a') as f:
        f.write(content + '\n')

def weighted_l1_loss(input, target, weights):
    """Calculate weighted L1 loss."""
    absolute_error = torch.abs(input - target)
    weighted_absolute_error = absolute_error * weights
    return weighted_absolute_error.mean()

def calculate_model_losses(model, input_tensors, arrays=None, arrays_p=None, v1=None, v2=None, scalars=None, v1_weight=None, v2_weight=None, device='cuda'):
    """
    Calculate losses exactly as in training for either encoder or decoder.
    
    Parameters:
        model: The model (encoder or decoder)
        input_tensors: Input tensors appropriate for the model
        arrays, arrays_p: Target arrays (for encoder)
        v1, v2, scalars: Input vectors and scalars (for decoder)
        v1_weight, v2_weight: Weights for vector reconstruction (for decoder)
        device: Device to use
    """
    if isinstance(model, EncoderUNet):
        # Encoder forward pass
        v1, v2, scalars = [t.to(device) for t in input_tensors]
        arrays = arrays.to(device) if arrays is not None else None
        outputs = model(v1, v2, scalars)
        
        # Forward loss calculation
        l1_loss_per_element = F.l1_loss(outputs, arrays, reduction='none')
        l1_loss_per_sample = l1_loss_per_element.sum(dim=[2, 3]).squeeze()
        weight = 1/scalars[:,0,0]
        loss_for = (l1_loss_per_sample * weight).mean()
        
        return {
            'forward_loss': loss_for.item(),
            'total_loss': loss_for.item()
        }
    else:
        # Decoder forward pass
        arrays_con = input_tensors.to(device)
        
        # Move all tensors to device and squeeze if needed
        v1 = v1.to(device) if v1 is not None else None
        v2 = v2.to(device) if v2 is not None else None
        scalars = scalars.to(device) if scalars is not None else None
        v1_weight = v1_weight.to(device) if v1_weight is not None else None
        v2_weight = v2_weight.to(device) if v2_weight is not None else None
        
        # Squeeze dimensions
        v1 = v1.squeeze(1) if v1 is not None else None
        v2 = v2.squeeze(1) if v2 is not None else None
        v1_weight = v1_weight.squeeze(1) if v1_weight is not None else None
        v2_weight = v2_weight.squeeze(1) if v2_weight is not None else None
        scalars = scalars.squeeze(1) if scalars is not None else None
        
        v1_reconstructed, v2_reconstructed, scalars_reconstructed = model(arrays_con)
        
        # Penalty loss
        penalty_loss = torch.where(v2_reconstructed < v1_reconstructed,
                                 v1_reconstructed - v2_reconstructed,
                                 torch.zeros_like(v2_reconstructed)).sum()
        
        # Reconstruction losses
        mse_loss_v1 = weighted_l1_loss(v1_reconstructed, v1, v1_weight)
        mse_loss_v2 = weighted_l1_loss(v2_reconstructed, v2, v2_weight)
        mse_loss_scalars = F.mse_loss(scalars_reconstructed, scalars) * 5
        
        loss_back = (mse_loss_v1 + mse_loss_v2 + mse_loss_scalars + penalty_loss * 10)
        
        return {
            'reconstruction_loss_v1': mse_loss_v1.item(),
            'reconstruction_loss_v2': mse_loss_v2.item(),
            'reconstruction_loss_scalars': mse_loss_scalars.item(),
            'penalty_loss': penalty_loss.item(),
            'total_loss': loss_back.item()
        }

def analyze_model_without_dataloader(model, input_tensors, arrays=None, arrays_p=None, v1=None, v2=None, scalars=None, v1_weight=None, v2_weight=None, device='cuda', model_name=""):
    """
    Analyzes a PyTorch model's weights, gradients, activations, and outputs.
    
    Parameters:
        model (torch.nn.Module): The model to analyze
        input_tensors: Input tensors appropriate for the model
        arrays, arrays_p: Target arrays (for encoder)
        v1, v2, scalars: Input vectors and scalars (for decoder)
        v1_weight, v2_weight: Weights for vector reconstruction (for decoder)
        device (str): Device to use ('cuda' or 'cpu')
        model_name (str): Name of the model for saving plots
    """
    model.to(device)
    model.eval()
    
    # Create directory for saving plots and analysis
    import os
    save_dir = f'analysis_results/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create or clear the analysis text file
    analysis_file = f"{save_dir}/analysis_report.txt"
    with open(analysis_file, 'w') as f:
        f.write(f"Analysis Report for {model_name}\n")
        f.write("=" * 50 + "\n\n")

    # Calculate losses
    losses = calculate_model_losses(
        model=model,
        input_tensors=input_tensors,
        arrays=arrays,
        arrays_p=arrays_p,
        v1=v1,
        v2=v2,
        scalars=scalars,
        v1_weight=v1_weight,
        v2_weight=v2_weight,
        device=device
    )
    
    # Save loss information
    save_to_file("\nLoss Analysis:", analysis_file)
    save_to_file("-" * 50, analysis_file)
    for loss_name, loss_value in losses.items():
        save_to_file(f"{loss_name}: {loss_value:.6f}", analysis_file)

    # Move input tensors to device
    if isinstance(model, EncoderUNet):
        v1, v2, scalars = [t.to(device) for t in input_tensors]
        arrays = arrays.to(device) if arrays is not None else None
    else:  # Decoder
        arrays_con = input_tensors.to(device)
        if v1 is not None: v1 = v1.to(device)
        if v2 is not None: v2 = v2.to(device)
        if scalars is not None: scalars = scalars.to(device)

    # 1. Analyze Weight Magnitudes and Distributions
    save_to_file(f"\nWeight Analysis for {model_name}:", analysis_file)
    save_to_file("-" * 50, analysis_file)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'max': param.data.max().item(),
                'min': param.data.min().item(),
                'shape': list(param.data.shape),
                'num_params': param.numel()
            }
            
            total_params += stats['num_params']
            if param.requires_grad:
                trainable_params += stats['num_params']
            
            save_to_file(f"\nLayer: {name}", analysis_file)
            save_to_file(f"Shape: {stats['shape']}", analysis_file)
            save_to_file(f"Parameters: {stats['num_params']:,}", analysis_file)
            save_to_file(f"Mean: {stats['mean']:.6f}", analysis_file)
            save_to_file(f"Std: {stats['std']:.6f}", analysis_file)
            save_to_file(f"Max: {stats['max']:.6f}", analysis_file)
            save_to_file(f"Min: {stats['min']:.6f}", analysis_file)
            
            plt.figure(figsize=(10, 6))
            plt.hist(param.data.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"Weight Distribution: {name}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"{save_dir}/weight_dist_{name.replace('.', '_')}.png")
            plt.close()
    
    save_to_file(f"\nTotal Parameters: {total_params:,}", analysis_file)
    save_to_file(f"Trainable Parameters: {trainable_params:,}", analysis_file)

    # 2. Forward Pass and Activation Analysis
    activations = {}

    def activation_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach() if not isinstance(output, tuple) else output[0].detach()
        return hook

    # Register hooks for ReLU and Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d)):
            module.register_forward_hook(activation_hook(name))
    
    save_to_file("\nActivation Analysis:", analysis_file)
    save_to_file("-" * 50, analysis_file)
    
    with torch.no_grad():
        if isinstance(model, EncoderUNet):
            output = model(v1, v2, scalars)
        else:  # Decoder
            output = model(arrays_con)
    
    # Analyze activations
    for name, activation in activations.items():
        stats = {
            'mean': activation.mean().item(),
            'std': activation.std().item(),
            'max': activation.max().item(),
            'min': activation.min().item(),
            'shape': list(activation.shape)
        }
        
        save_to_file(f"\nLayer: {name}", analysis_file)
        save_to_file(f"Shape: {stats['shape']}", analysis_file)
        save_to_file(f"Mean: {stats['mean']:.6f}", analysis_file)
        save_to_file(f"Std: {stats['std']:.6f}", analysis_file)
        save_to_file(f"Max: {stats['max']:.6f}", analysis_file)
        save_to_file(f"Min: {stats['min']:.6f}", analysis_file)
        
        if isinstance(model.get_submodule(name), nn.ReLU):
            dead_units = (activation == 0).float().mean().item()
            save_to_file(f"Dead Units: {dead_units * 100:.2f}%", analysis_file)
        
        plt.figure(figsize=(10, 6))
        plt.hist(activation.detach().cpu().numpy().flatten(), bins=50)
        plt.title(f"Activation Distribution: {name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f"{save_dir}/activation_dist_{name.replace('.', '_')}.png")
        plt.close()

    # 3. Gradient Analysis
    save_to_file("\nGradient Analysis:", analysis_file)
    save_to_file("-" * 50, analysis_file)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    
    if isinstance(model, EncoderUNet):
        output = model(v1, v2, scalars)
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)
    else:  # Decoder
        v1_reconstructed, v2_reconstructed, scalars_reconstructed = model(arrays_con)
        # Calculate separate losses for each output
        loss_v1 = nn.MSELoss()(v1_reconstructed, torch.zeros_like(v1_reconstructed))
        loss_v2 = nn.MSELoss()(v2_reconstructed, torch.zeros_like(v2_reconstructed))
        loss_scalars = nn.MSELoss()(scalars_reconstructed, torch.zeros_like(scalars_reconstructed))
        loss = loss_v1 + loss_v2 + loss_scalars
        
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            stats = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item(),
                'norm': param.grad.norm().item()
            }
            
            save_to_file(f"\nLayer: {name}", analysis_file)
            save_to_file(f"Gradient Mean: {stats['mean']:.6f}", analysis_file)
            save_to_file(f"Gradient Std: {stats['std']:.6f}", analysis_file)
            save_to_file(f"Gradient Max: {stats['max']:.6f}", analysis_file)
            save_to_file(f"Gradient Min: {stats['min']:.6f}", analysis_file)
            save_to_file(f"Gradient Norm: {stats['norm']:.6f}", analysis_file)
            
            plt.figure(figsize=(10, 6))
            plt.hist(param.grad.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"Gradient Distribution: {name}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"{save_dir}/gradient_dist_{name.replace('.', '_')}.png")
            plt.close()

    # 4. Output Analysis
    save_to_file("\nOutput Analysis:", analysis_file)
    save_to_file("-" * 50, analysis_file)
    
    if isinstance(output, tuple):
        # For decoder output
        for i, out in enumerate(['v1', 'v2', 'scalars']):
            stats = {
                'mean': output[i].mean().item(),
                'std': output[i].std().item(),
                'max': output[i].max().item(),
                'min': output[i].min().item(),
                'shape': list(output[i].shape)
            }
            
            save_to_file(f"\nOutput {out}:", analysis_file)
            save_to_file(f"Shape: {stats['shape']}", analysis_file)
            save_to_file(f"Mean: {stats['mean']:.6f}", analysis_file)
            save_to_file(f"Std: {stats['std']:.6f}", analysis_file)
            save_to_file(f"Max: {stats['max']:.6f}", analysis_file)
            save_to_file(f"Min: {stats['min']:.6f}", analysis_file)
            
            plt.figure(figsize=(10, 6))
            plt.hist(output[i].detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"Output Distribution: {out}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"{save_dir}/output_dist_{out}.png")
            plt.close()
    else:
        # For encoder output
        stats = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'max': output.max().item(),
            'min': output.min().item(),
            'shape': list(output.shape)
        }
        
        save_to_file("\nOutput:", analysis_file)
        save_to_file(f"Shape: {stats['shape']}", analysis_file)
        save_to_file(f"Mean: {stats['mean']:.6f}", analysis_file)
        save_to_file(f"Std: {stats['std']:.6f}", analysis_file)
        save_to_file(f"Max: {stats['max']:.6f}", analysis_file)
        save_to_file(f"Min: {stats['min']:.6f}", analysis_file)
        
        plt.figure(figsize=(10, 6))
        plt.hist(output.detach().cpu().numpy().flatten(), bins=50)
        plt.title("Output Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f"{save_dir}/output_distribution.png")
        plt.close()

    save_to_file(f"\nAnalysis Complete for {model_name}!", analysis_file)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained models
    checkpoint_path = 'Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    vector_dim = 104
    scalar_count = 5
    latent_image_size = 128
    in_channels = 1
    out_channels = 1
    resize_out = 131
    resize_in = 128
    
    # Initialize encoder
    encoderunet = EncoderUNet(ExtEncoder, vector_dim, scalar_count, latent_image_size,
                             in_channels, out_channels, resize_out)
    encoderunet_state = {k.replace('module.', ''): v for k, v in checkpoint['encoderunet_state_dict'].items()}
    encoderunet.load_state_dict(encoderunet_state)
    encoderunet.to(device)
    
    # Initialize decoder
    unetdecoder = UNetDecoder(ExtDecoder, vector_dim, scalar_count, latent_image_size,
                             2, out_channels, resize_in)
    unetdecoder_state = {k.replace('module.', ''): v for k, v in checkpoint['unetdecoder_state_dict'].items()}
    unetdecoder.load_state_dict(unetdecoder_state)
    unetdecoder.to(device)
    
    # Create sample input tensors (all on the specified device)
    batch_size = 32
    
    # Input for encoder
    v1 = torch.randn(batch_size, 1, vector_dim, device=device)
    v2 = torch.randn(batch_size, 1, vector_dim, device=device)
    scalars = torch.randn(batch_size, 1, scalar_count, device=device)
    arrays = torch.randn(batch_size, 1, 131, 131, device=device)  # Target arrays
    arrays_p = torch.randn(batch_size, 1, 131, 131, device=device)  # Previous arrays
    
    # Input for decoder
    arrays_con = torch.randn(batch_size, 2, 131, 131, device=device)  # Concatenated arrays
    v1_weight = torch.ones(batch_size, 1, vector_dim, device=device)  # Vector weights
    v2_weight = torch.ones(batch_size, 1, vector_dim, device=device)
    
    # Analyze encoder
    print("\nAnalyzing Encoder-UNet...")
    analyze_model_without_dataloader(
        model=encoderunet,
        input_tensors=(v1, v2, scalars),
        arrays=arrays,
        arrays_p=arrays_p,
        device=device,
        model_name="encoder"
    )
    
    # Analyze decoder
    print("\nAnalyzing Decoder-UNet...")
    analyze_model_without_dataloader(
        model=unetdecoder,
        input_tensors=arrays_con,
        v1=v1,
        v2=v2,
        scalars=scalars,
        v1_weight=v1_weight,
        v2_weight=v2_weight,
        device=device,
        model_name="decoder"
    )

if __name__ == "__main__":
    main() 