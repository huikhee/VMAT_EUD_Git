# VMAT Machine Learning Project

This project implements machine learning models for Volumetric Modulated Arc Therapy (VMAT) using PyTorch. It includes both encoder and decoder networks to process and generate VMAT treatment plans.

## Project Structure

```
src/vmat_ml/
├── models/
│   ├── common.py         # Common model components
│   ├── encoder_unet.py   # Encoder-UNet model
│   └── unet_decoder.py   # UNet-Decoder model
├── data/
│   ├── generators.py     # Data generation utilities
│   └── datasets.py       # PyTorch dataset classes
├── utils/
│   ├── metrics.py        # Evaluation metrics
│   └── visualization.py  # Visualization tools
├── training/
│   └── training.py       # Training functions
├── train.py             # Main training script
└── inference.py         # Inference script
```

## Features

- Encoder-UNet model for generating VMAT frames from MLC positions and parameters
- UNet-Decoder model for reconstructing MLC positions from VMAT frames
- Support for both synthetic and real VMAT data
- Gamma index calculation for model evaluation
- Comprehensive visualization tools
- Training with early stopping and learning rate scheduling
- Inference pipeline for trained models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vmat_ml.git
cd vmat_ml
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the models, use the `train.py` script. You can train either the encoder, decoder, or both:

```bash
# Train both models
python -m vmat_ml.train --train-encoder --train-decoder --epochs 100 --batch-size 32 --learning-rate 0.001

# Train only encoder
python -m vmat_ml.train --train-encoder --epochs 100

# Train only decoder
python -m vmat_ml.train --train-decoder --epochs 100
```

Training parameters:
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs (default: 100)
- `--learning-rate`: Learning rate (default: 0.001)
- `--latent-size`: Size of latent image (default: 32)
- `--model-dir`: Directory to save models (default: 'models')
- `--no-cuda`: Disable CUDA training
- `--seed`: Random seed (default: 42)

### Inference

To run inference with trained models:

```bash
python -m vmat_ml.inference \
    --model-dir models \
    --test-data-path path/to/test/data.npz \
    --num-samples 5
```

Inference parameters:
- `--model-dir`: Directory containing trained models (required)
- `--test-data-path`: Path to test data file (required)
- `--latent-size`: Size of latent image (default: 32)
- `--num-samples`: Number of samples to visualize (default: 5)
- `--no-cuda`: Disable CUDA

## Model Architecture

### Encoder-UNet
- ExtEncoder: Processes MLC positions and parameters into latent images
- UNet: Generates VMAT frames from latent images
- Features skip connections and optional layer freezing

### UNet-Decoder
- UNet2: Processes VMAT frames
- ExtDecoder: Reconstructs MLC positions and parameters
- Supports both training and inference modes

## Data

The project supports two types of data:
1. Synthetic data generated using the functions in `generators.py`
2. Real VMAT data loaded through the `RealVMATDataset` class

## Evaluation

Models are evaluated using:
- Mean Squared Error (MSE) loss
- Gamma index calculation
- Visualization tools for comparing predictions with ground truth

## Visualization

The visualization module provides:
- Actual vs. predicted frame comparison
- Gamma index visualization
- MLC position plots
- Monitor unit comparisons

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm
- pymedphys (for gamma index calculation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 