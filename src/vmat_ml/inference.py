import torch
import numpy as np
import argparse
import os
from models.encoder_unet import EncoderUNet
from models.unet_decoder import UNetDecoder
from data.datasets import RealVMATDataset
from utils.visualization import visualize_results
from torch.utils.data import DataLoader

def load_model(model_path, model_class, model_params, device):
    """Load a trained model.
    
    Args:
        model_path (str): Path to the model weights
        model_class: Model class to instantiate
        model_params (dict): Parameters for model initialization
        device: PyTorch device
        
    Returns:
        model: Loaded model
    """
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(encoder_unet, unet_decoder, dataloader, device, num_samples=5):
    """Run inference on a batch of data.
    
    Args:
        encoder_unet: Trained encoder-unet model
        unet_decoder: Trained unet-decoder model
        dataloader: DataLoader containing the test data
        device: PyTorch device
        num_samples (int): Number of samples to visualize
    """
    # Get a batch of data
    data_iter = iter(dataloader)
    for i in range(min(num_samples, len(dataloader))):
        try:
            sample = next(data_iter)
            visualize_results(i, dataloader.dataset, encoder_unet, unet_decoder, device)
        except StopIteration:
            break

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    # Load test data
    test_data = np.load(args.test_data_path)
    dataset = RealVMATDataset(
        segmentMU=test_data['segmentMU'],
        mlcx1=test_data['mlcx1'],
        mlcx2=test_data['mlcx2'],
        y1=test_data['y1'],
        y2=test_data['y2'],
        accumulated_frames=test_data['accumulated_frames']
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model parameters
    model_params = {
        'vector_dim': 52,
        'scalar_count': 5,
        'latent_image_size': args.latent_size,
        'in_channels': 1,
        'out_channels': 1,
        'resize_out': 261,
        'freeze_encoder': False
    }

    # Load models
    encoder_unet = load_model(
        model_path=os.path.join(args.model_dir, 'encoder_unet.pth'),
        model_class=EncoderUNet,
        model_params=model_params,
        device=device
    )

    decoder_params = model_params.copy()
    decoder_params['in_channels'] = 2
    decoder_params['resize_in'] = 261
    unet_decoder = load_model(
        model_path=os.path.join(args.model_dir, 'unet_decoder.pth'),
        model_class=UNetDecoder,
        model_params=decoder_params,
        device=device
    )

    # Run inference
    print('Running inference...')
    run_inference(encoder_unet, unet_decoder, dataloader, device, args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained VMAT models')
    
    # Model settings
    parser.add_argument('--model-dir', type=str, required=True, help='directory containing trained models')
    parser.add_argument('--test-data-path', type=str, required=True, help='path to test data file')
    parser.add_argument('--latent-size', type=int, default=32, help='size of latent image (default: 32)')
    parser.add_argument('--num-samples', type=int, default=5, help='number of samples to visualize (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    
    args = parser.parse_args()
    main(args) 