"""Shared-latent EPID training pipeline.

This script reuses the existing VMAT synthetic data format but couples the
parameter-to-image and image-to-parameter tasks through a single latent
representation. Both directions share the same latent vector so gradients
from EPID synthesis and inverse reconstruction reinforce one another.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# macOS sometimes fails to locate libgfortran/libopenblas inside conda envs.
# Prepend the env's lib directory to DYLD_LIBRARY_PATH before importing numpy.
_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _conda_lib not in _dyld.split(":"):
        os.environ["DYLD_LIBRARY_PATH"] = (
            f"{_conda_lib}:{_dyld}" if _dyld else _conda_lib
        )

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

#############################################
# Dataset utilities
#############################################

class CustomDataset(Dataset):
    """Matches the dataset layout used in generate_and_train_EPID_v1.py."""

    def __init__(self, vector1, vector2, scalar1, scalar2, scalar3,
                 vector1_weight, vector2_weight, arrays):
        self.vector1 = torch.from_numpy(vector1).float()
        self.vector2 = torch.from_numpy(vector2).float()
        self.scalar1 = torch.from_numpy(scalar1).float()
        self.scalar2 = torch.from_numpy(scalar2).float()
        self.scalar3 = torch.from_numpy(scalar3).float()
        self.vector1_weight = torch.from_numpy(vector1_weight).float()
        self.vector2_weight = torch.from_numpy(vector2_weight).float()
        self.arrays = torch.from_numpy(arrays).float()

    def __len__(self) -> int:
        return len(self.vector1) - 2

    def __getitem__(self, idx: int):
        idx += 2
        prev_idx = idx - 1

        v1 = torch.cat(
            [self.vector1[prev_idx].unsqueeze(0), self.vector1[idx].unsqueeze(0)],
            dim=1,
        )
        v2 = torch.cat(
            [self.vector2[prev_idx].unsqueeze(0), self.vector2[idx].unsqueeze(0)],
            dim=1,
        )
        v1_weight = torch.cat(
            [self.vector1_weight[prev_idx].unsqueeze(0), self.vector1_weight[idx].unsqueeze(0)],
            dim=1,
        )
        v2_weight = torch.cat(
            [self.vector2_weight[prev_idx].unsqueeze(0), self.vector2_weight[idx].unsqueeze(0)],
            dim=1,
        )

        scalar1 = self.scalar1[idx].unsqueeze(0).unsqueeze(0)
        scalar2_current = self.scalar2[idx].unsqueeze(0).unsqueeze(0)
        scalar2_previous = self.scalar2[prev_idx].unsqueeze(0).unsqueeze(0)
        scalar3_current = self.scalar3[idx].unsqueeze(0).unsqueeze(0)
        scalar3_previous = self.scalar3[prev_idx].unsqueeze(0).unsqueeze(0)

        scalars = torch.cat(
            [scalar1, scalar2_previous, scalar2_current, scalar3_previous, scalar3_current],
            dim=1,
        )

        arrays = self.arrays[idx].unsqueeze(0)
        arrays_p = self.arrays[prev_idx].unsqueeze(0)

        return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p


def dataset_path(dataset_num: int) -> Path:
    return Path("VMAT_Art_data") / f"Art_dataset_coll0_{dataset_num}.pt"


def load_dataset(dataset_num: int) -> Optional[Dataset]:
    path = dataset_path(dataset_num)
    if path.exists():
        return torch.load(path, map_location="cpu")
    return None


def ensure_dataset(dataset_num: int, KM: Optional[torch.Tensor]):
    existing = load_dataset(dataset_num)
    if existing is not None:
        return existing
    if KM is None:
        raise RuntimeError("KM kernel required to regenerate datasets.")
    from generate_and_train_EPID_v1 import generate_and_save_dataset  # lazy import

    return generate_and_save_dataset(dataset_num, KM)


#############################################
# Shared latent model components
#############################################

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class ParameterEncoder(nn.Module):
    def __init__(self, vector_dim: int, scalar_count: int, latent_dim: int):
        super().__init__()
        self.vector_fc = nn.Sequential(
            nn.Linear(vector_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.scalar_emb = nn.ModuleList([nn.Linear(1, 32) for _ in range(scalar_count)])
        self.scalar_norm = nn.LayerNorm(scalar_count * 32)
        self.head = nn.Sequential(
            nn.Linear(256 + scalar_count * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, vector1: torch.Tensor, vector2: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        bsz = vector1.size(0)
        vector1 = vector1.view(bsz, -1)
        vector2 = vector2.view(bsz, -1)
        vec_features = self.vector_fc(torch.cat([vector1, vector2], dim=1))
        if scalars.dim() == 3 and scalars.size(1) == 1:
            scalars = scalars.squeeze(1)
        scalar_features = [emb(scalars[:, i].unsqueeze(1)) for i, emb in enumerate(self.scalar_emb)]
        scalar_features = torch.cat(scalar_features, dim=1)
        scalar_features = self.scalar_norm(scalar_features).relu_()
        return self.head(torch.cat([vec_features, scalar_features], dim=1))


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.resize = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False)
        self.encoder1 = EncoderBlock(2, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.bottleneck = ResidualConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, stacked_images: torch.Tensor) -> torch.Tensor:
        x = self.resize(stacked_images)
        _, p1 = self.encoder1(x)
        _, p2 = self.encoder2(p1)
        _, p3 = self.encoder3(p2)
        btl = self.bottleneck(p3)
        pooled = self.global_pool(btl).view(btl.size(0), -1)
        return self.fc(pooled)


class EPIDDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.GroupNorm(1, 16),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(16, 1, 3, padding=1)
        self.final_resize = nn.Upsample(size=(131, 131), mode="bilinear", align_corners=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent).view(latent.size(0), 256, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        return self.final_resize(torch.relu(x))


class ParameterDecoder(nn.Module):
    def __init__(self, vector_dim: int, scalar_count: int, latent_dim: int):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        self.vector1_head = nn.Linear(512, vector_dim)
        self.vector2_head = nn.Linear(512, vector_dim)
        self.scalar_head = nn.Linear(512, scalar_count)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.core(latent)
        return self.vector1_head(features), self.vector2_head(features), self.scalar_head(features)


class SharedLatentBidirectionalModel(nn.Module):
    def __init__(self, vector_dim: int, scalar_count: int, latent_dim: int):
        super().__init__()
        self.param_encoder = ParameterEncoder(vector_dim, scalar_count, latent_dim)
        self.image_encoder = ImageEncoder(latent_dim)
        self.epid_decoder = EPIDDecoder(latent_dim)
        self.param_decoder = ParameterDecoder(vector_dim, scalar_count, latent_dim)

    def forward(self, vector1: torch.Tensor, vector2: torch.Tensor, scalars: torch.Tensor,
                stacked_images: torch.Tensor) -> dict:
        latent_params = self.param_encoder(vector1, vector2, scalars)
        latent_images = self.image_encoder(stacked_images)
        epid_from_params = self.epid_decoder(latent_params)
        epid_from_images = self.epid_decoder(latent_images)
        v1_from_image, v2_from_image, scalars_from_image = self.param_decoder(latent_images)
        v1_from_param, v2_from_param, scalars_from_param = self.param_decoder(latent_params)
        return {
            "latent_params": latent_params,
            "latent_images": latent_images,
            "epid_from_params": epid_from_params,
            "epid_from_images": epid_from_images,
            "v1_from_image": v1_from_image,
            "v2_from_image": v2_from_image,
            "scalars_from_image": scalars_from_image,
            "v1_from_param": v1_from_param,
            "v2_from_param": v2_from_param,
            "scalars_from_param": scalars_from_param,
        }


#############################################
# Loss helpers
#############################################

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def separation_penalty(v1_pred: torch.Tensor, v2_pred: torch.Tensor) -> torch.Tensor:
    return torch.relu(v1_pred - v2_pred).mean()


#############################################
# Training and evaluation
#############################################

def move_batch_to_device(batch: Sequence[torch.Tensor], device: torch.device):
    return [tensor.to(device, non_blocking=True) for tensor in batch]


def prepare_image_stack(arrays: torch.Tensor, arrays_p: torch.Tensor) -> torch.Tensor:
    arrays = arrays.squeeze(1)
    arrays_p = arrays_p.squeeze(1)
    return torch.cat([arrays_p, arrays], dim=1)


def compute_losses(batch_outputs: dict, batch_inputs: dict, config: dict) -> Tuple[torch.Tensor, dict]:
    arrays = batch_inputs["arrays"]
    scalars = batch_inputs["scalars"].squeeze(1)
    v1 = batch_inputs["v1"].squeeze(1)
    v2 = batch_inputs["v2"].squeeze(1)
    v1_weight = batch_inputs["v1_weight"].squeeze(1)
    v2_weight = batch_inputs["v2_weight"].squeeze(1)

    image_weights = (1.0 / scalars[:, 0]).view(-1, 1, 1, 1).expand_as(arrays)

    loss_epid_forward = weighted_mse_loss(batch_outputs["epid_from_params"], arrays, image_weights)
    loss_epid_cycle = weighted_mse_loss(batch_outputs["epid_from_images"], arrays, image_weights)

    loss_v1_image = weighted_mse_loss(batch_outputs["v1_from_image"], v1, v1_weight)
    loss_v2_image = weighted_mse_loss(batch_outputs["v2_from_image"], v2, v2_weight)
    loss_scalar_image = mse_loss(batch_outputs["scalars_from_image"], scalars)

    loss_v1_cycle = weighted_mse_loss(batch_outputs["v1_from_param"], v1, v1_weight)
    loss_v2_cycle = weighted_mse_loss(batch_outputs["v2_from_param"], v2, v2_weight)
    loss_scalar_cycle = mse_loss(batch_outputs["scalars_from_param"], scalars)

    penalty_image = separation_penalty(batch_outputs["v1_from_image"], batch_outputs["v2_from_image"])
    penalty_cycle = separation_penalty(batch_outputs["v1_from_param"], batch_outputs["v2_from_param"])

    latent_align = mse_loss(batch_outputs["latent_params"], batch_outputs["latent_images"])

    total_loss = (
        config["lambda_epid"] * loss_epid_forward
        + config["lambda_epid_cycle"] * loss_epid_cycle
        + config["lambda_param"] * (loss_v1_image + loss_v2_image + loss_scalar_image)
        + config["lambda_param_cycle"] * (loss_v1_cycle + loss_v2_cycle + loss_scalar_cycle)
        + config["lambda_penalty"] * (penalty_image + penalty_cycle)
        + config["lambda_latent"] * latent_align
    )

    metrics = {
        "loss": total_loss.detach(),
        "loss_epid": loss_epid_forward.detach(),
        "loss_epid_cycle": loss_epid_cycle.detach(),
        "loss_param": (loss_v1_image + loss_v2_image + loss_scalar_image).detach(),
        "loss_param_cycle": (loss_v1_cycle + loss_v2_cycle + loss_scalar_cycle).detach(),
        "latent_align": latent_align.detach(),
    }
    return total_loss, metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device, config: dict) -> dict:
    model.train()
    aggregate = {"loss": 0.0}
    for batch in loader:
        v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p = move_batch_to_device(batch, device)
        stacked_images = prepare_image_stack(arrays, arrays_p)
        batch_inputs = {
            "arrays": arrays,
            "scalars": scalars,
            "v1": v1,
            "v2": v2,
            "v1_weight": v1_weight,
            "v2_weight": v2_weight,
        }
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config["amp"]):
            outputs = model(v1, v2, scalars, stacked_images)
            loss, metrics = compute_losses(outputs, batch_inputs, config)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        aggregate["loss"] += metrics["loss"].item()
    aggregate["loss"] /= max(1, len(loader))
    return aggregate


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: dict) -> dict:
    model.eval()
    aggregate = {"loss": 0.0}
    with torch.no_grad(), autocast(enabled=config["amp"]):
        for batch in loader:
            v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p = move_batch_to_device(batch, device)
            stacked_images = prepare_image_stack(arrays, arrays_p)
            batch_inputs = {
                "arrays": arrays,
                "scalars": scalars,
                "v1": v1,
                "v2": v2,
                "v1_weight": v1_weight,
                "v2_weight": v2_weight,
            }
            outputs = model(v1, v2, scalars, stacked_images)
            loss, _ = compute_losses(outputs, batch_inputs, config)
            aggregate["loss"] += loss.item()
    aggregate["loss"] /= max(1, len(loader))
    return aggregate


#############################################
# Data loading helpers
#############################################

def build_dataloaders(dataset_indices: Iterable[int], batch_size: int, split: float,
                      seed: int, regenerate_missing: bool,
                      KM: Optional[torch.Tensor]) -> Tuple[DataLoader, DataLoader]:
    datasets: List[Dataset] = []
    for idx in dataset_indices:
        data = load_dataset(idx)
        if data is None:
            if regenerate_missing:
                data = ensure_dataset(idx, KM)
            else:
                print(f"[WARN] Dataset {idx} missing; skipping.")
                continue
        datasets.append(data)
    if not datasets:
        raise RuntimeError("No datasets available for training.")

    concat = ConcatDataset(datasets)
    train_size = int(len(concat) * split)
    val_size = len(concat) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(concat, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)
    return train_loader, val_loader


#############################################
# Main entry point
#############################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train shared-latent EPID model.")
    parser.add_argument("--dataset-range", type=str, default="0:32",
                        help="Range of dataset indices to use, e.g. 0:64")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate-missing", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="Cross_CP/shared_latent_checkpoint.pth")
    return parser.parse_args()


def parse_range(text: str) -> Sequence[int]:
    if ":" in text:
        start, end = text.split(":", 1)
        return range(int(start), int(end))
    return [int(text)]


def maybe_load_KM(regenerate_missing: bool):
    if not regenerate_missing:
        return None
    if loadmat is None:
        raise RuntimeError("scipy is required to regenerate datasets.")
    data = loadmat("data/KM_1500.mat")
    return torch.tensor(data["KM_1500"]).float()


def main():
    args = parse_args()
    os.makedirs("Cross_CP", exist_ok=True)
    os.makedirs("VMAT_Art_data", exist_ok=True)

    dataset_indices = list(parse_range(args.dataset_range))
    KM = maybe_load_KM(args.regenerate_missing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    vector_dim = 104
    scalar_count = 5

    config = {
        "lambda_epid": 1.0,
        "lambda_epid_cycle": 0.5,
        "lambda_param": 1.0,
        "lambda_param_cycle": 0.5,
        "lambda_penalty": 10.0,
        "lambda_latent": 0.1,
        "amp": device.type == "cuda",
    }

    train_loader, val_loader = build_dataloaders(
        dataset_indices,
        batch_size=args.batch_size,
        split=args.split,
        seed=args.seed,
        regenerate_missing=args.regenerate_missing,
        KM=KM,
    )

    model = SharedLatentBidirectionalModel(vector_dim, scalar_count, args.latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=config["amp"])

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        print(f"Loaded checkpoint from {checkpoint_path} at epoch {state['epoch']}")
        start_epoch = state["epoch"] + 1
    else:
        start_epoch = 0

    best_val = math.inf

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        val_metrics = evaluate(model, val_loader, device, config)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:04d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4e} | "
            f"Val Loss: {val_metrics['loss']:.4e} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
