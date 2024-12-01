import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ..utils.metrics import calculate_gamma_index

def train_encoder_unet(model, train_loader, val_loader, num_epochs, learning_rate, device, 
                      save_path=None, scheduler_patience=5, early_stopping_patience=10):
    """Train the encoder-unet model.
    
    Args:
        model: The encoder-unet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        device: PyTorch device
        save_path (str, optional): Path to save the best model
        scheduler_patience (int): Patience for learning rate scheduler
        early_stopping_patience (int): Patience for early stopping
        
    Returns:
        dict: Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_gamma': []
    }

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        for v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            v1, v2 = v1.to(device), v2.to(device)
            scalars = scalars.to(device)
            arrays = arrays.to(device)

            optimizer.zero_grad()
            outputs = model(v1, v2, scalars)
            loss = criterion(outputs, arrays)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        gamma_scores = []

        with torch.no_grad():
            for v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p in val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                scalars = scalars.to(device)
                arrays = arrays.to(device)

                outputs = model(v1, v2, scalars)
                loss = criterion(outputs, arrays)
                val_loss += loss.item()
                val_batches += 1

                # Calculate gamma index for each sample in batch
                for i in range(arrays.size(0)):
                    gamma_score = calculate_gamma_index(
                        arrays[i].squeeze().cpu(),
                        outputs[i].squeeze().cpu()
                    )
                    gamma_scores.append(gamma_score)

        avg_val_loss = val_loss / val_batches
        avg_gamma_score = np.mean(gamma_scores)
        
        history['val_loss'].append(avg_val_loss)
        history['val_gamma'].append(avg_gamma_score)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'Val Gamma Score: {avg_gamma_score:.2f}%')

        if early_stopping_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break

    return history

def train_unet_decoder(model, train_loader, val_loader, num_epochs, learning_rate, device,
                      save_path=None, scheduler_patience=5, early_stopping_patience=10):
    """Train the unet-decoder model.
    
    Args:
        model: The unet-decoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        device: PyTorch device
        save_path (str, optional): Path to save the best model
        scheduler_patience (int): Patience for learning rate scheduler
        early_stopping_patience (int): Patience for early stopping
        
    Returns:
        dict: Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        for v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            arrays = arrays.to(device)
            arrays_p = arrays_p.to(device)
            v1, v2 = v1.to(device), v2.to(device)
            scalars = scalars.to(device)

            optimizer.zero_grad()
            arrays_con = torch.cat([arrays_p, arrays], dim=1)
            v1_pred, v2_pred, scalars_pred = model(arrays_con)

            # Calculate losses for each component
            v1_loss = criterion(v1_pred, v1)
            v2_loss = criterion(v2_pred, v2)
            scalars_loss = criterion(scalars_pred, scalars[:, 0])

            # Combine losses
            loss = v1_loss + v2_loss + scalars_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p in val_loader:
                arrays = arrays.to(device)
                arrays_p = arrays_p.to(device)
                v1, v2 = v1.to(device), v2.to(device)
                scalars = scalars.to(device)

                arrays_con = torch.cat([arrays_p, arrays], dim=1)
                v1_pred, v2_pred, scalars_pred = model(arrays_con)

                v1_loss = criterion(v1_pred, v1)
                v2_loss = criterion(v2_pred, v2)
                scalars_loss = criterion(scalars_pred, scalars[:, 0])

                loss = v1_loss + v2_loss + scalars_loss
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')

        if early_stopping_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break

    return history 