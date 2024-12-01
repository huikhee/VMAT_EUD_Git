import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
from models.encoder_unet import EncoderUNet
from models.unet_decoder import UNetDecoder
from data.generators import generate_random_vectors_scalar_regular, create_boundary_matrix
from data.datasets import CustomDataset
from training.training import train_encoder_unet, train_unet_decoder

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    # Generate synthetic data
    print('Generating synthetic data...')
    vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight = generate_random_vectors_scalar_regular(args.seed)
    arrays = create_boundary_matrix(vector1, vector2, scalar1, scalar2, scalar3)

    # Create dataset
    dataset = CustomDataset(vector1, vector2, scalar1, scalar2, scalar3, vector1_weight, vector2_weight, arrays)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create models directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    if args.train_encoder:
        print('Training Encoder-UNet model...')
        encoder_unet = EncoderUNet(
            vector_dim=52,
            scalar_count=5,
            latent_image_size=args.latent_size,
            in_channels=1,
            out_channels=1,
            resize_out=261,
            freeze_encoder=False
        ).to(device)

        encoder_history = train_encoder_unet(
            model=encoder_unet,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            save_path=os.path.join(args.model_dir, 'encoder_unet.pth')
        )
        print('Encoder-UNet training completed')

    if args.train_decoder:
        print('Training UNet-Decoder model...')
        unet_decoder = UNetDecoder(
            vector_dim=52,
            scalar_count=5,
            latent_image_size=args.latent_size,
            in_channels=2,
            out_channels=1,
            resize_in=261,
            freeze_encoder=False
        ).to(device)

        decoder_history = train_unet_decoder(
            model=unet_decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            save_path=os.path.join(args.model_dir, 'unet_decoder.pth')
        )
        print('UNet-Decoder training completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VMAT models')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    
    # Model settings
    parser.add_argument('--latent-size', type=int, default=32, help='size of latent image (default: 32)')
    parser.add_argument('--model-dir', type=str, default='models', help='directory to save models')
    
    # Training mode
    parser.add_argument('--train-encoder', action='store_true', help='train encoder-unet model')
    parser.add_argument('--train-decoder', action='store_true', help='train unet-decoder model')
    
    args = parser.parse_args()
    
    if not args.train_encoder and not args.train_decoder:
        parser.error('At least one of --train-encoder or --train-decoder must be specified')
    
    main(args) 