"""Inference + visualization for the EPID v5 encoder/decoder model.

v5 changes (relative to v4) are in the *backward* path (UNetDecoder transformer head):
- (1) Typed scalar queries: MU + jaw1 + jaw2
- (2) Leaf decoding as (prev,curr) pairs: 52 tokens per bank, repacked to 104

Forward path (EncoderUNet) remains v4 tokenization.

This script mirrors v4 inference but imports v5 models and updates defaults.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Iterable, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_and_train_EPID_v5 import (
    CustomDataset,
    EncoderUNet,
    UNetDecoder,
    generate_and_save_dataset,
    load_dataset,
)


plt.switch_backend("Agg")  # headless safe


def parse_range(spec: str) -> Iterable[int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("range must be start:end or start:end:step")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    return range(start, end, step)


def load_or_generate_dataset(idx: int, regenerate_missing: bool, KM: np.ndarray | None) -> CustomDataset:
    ds = load_dataset(idx)
    if ds is None:
        if regenerate_missing:
            if KM is None:
                raise RuntimeError("KM kernel required to regenerate dataset")
            ds = generate_and_save_dataset(idx, KM)
            print(f"[INFO] Generated dataset {idx}")
        else:
            raise RuntimeError(f"Dataset {idx} missing; rerun with --regenerate-missing")
    return ds  # type: ignore[return-value]


def strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def ensure_nchw_epid(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 5 and t.size(2) == 1:
        t = t.squeeze(2)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def compute_gamma(epid_gt: np.ndarray, epid_pred: np.ndarray) -> Tuple[np.ndarray | None, float | None]:
    """Compute a 2D gamma map + pass rate using pymedphys if available."""
    try:
        import pymedphys  # type: ignore
    except Exception:
        return None, None

    ref = np.asarray(epid_gt, dtype=float).squeeze()
    eva = np.asarray(epid_pred, dtype=float).squeeze()
    if ref.ndim != 2 or eva.ndim != 2 or ref.shape != eva.shape:
        return None, None

    y = np.linspace(-130.0, 130.0, ref.shape[0])
    x = np.linspace(-130.0, 130.0, ref.shape[1])
    axes = (y, x)

    try:
        gamma_map = pymedphys.gamma(
            axes,
            ref,
            axes,
            eva,
            dose_percent_threshold=3,
            distance_mm_threshold=3,
            interp_fraction=10,
            max_gamma=2,
            quiet=True,
        )
    except Exception:
        return None, None

    valid = np.isfinite(gamma_map)
    if not np.any(valid):
        return gamma_map, None
    pass_rate = float(100.0 * np.mean((gamma_map[valid] <= 1.0)))
    return gamma_map, pass_rate


def plot_losses(train_losses, val_losses, out_path: Path) -> None:
    if not train_losses and not val_losses:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss", color="tab:blue")
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss", color="tab:orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training/Validation Loss (log scale)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if train_losses and val_losses:
        ax.legend()

    min_val = None
    for arr in (train_losses, val_losses):
        for v in arr:
            if min_val is None or v < min_val:
                min_val = v
    if min_val is not None and min_val <= 0:
        shift = abs(min_val) + 1e-8
        train_losses = [v + shift for v in train_losses]
        val_losses = [v + shift for v in val_losses]
        ax.clear()
        if train_losses:
            ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss", color="tab:blue")
        if val_losses:
            ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss (shifted by {shift:.2e})")
        ax.set_title("Training/Validation Loss (log scale)")
        ax.grid(True, linestyle="--", alpha=0.4)
        if train_losses and val_losses:
            ax.legend()

    ax.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss plot saved to {out_path}")


COLORS = ["green", "red"]
CMAP = mcolors.ListedColormap(COLORS)
NORM = mcolors.BoundaryNorm([0, 1, 2], len(COLORS), clip=True)


def visualize(
    epid_gt: np.ndarray,
    epid_pred: np.ndarray,
    v1_gt: np.ndarray,
    v2_gt: np.ndarray,
    v1_pred: np.ndarray,
    v2_pred: np.ndarray,
    scalars_gt: np.ndarray,
    scalars_pred: np.ndarray,
    v1_weight: np.ndarray,
    v2_weight: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    epid_gt_np = np.squeeze(epid_gt)
    epid_pred_np = np.squeeze(epid_pred)

    v1_gt_np = v1_gt.reshape(-1)
    v2_gt_np = v2_gt.reshape(-1)
    v1_pred_np = v1_pred.reshape(-1)
    v2_pred_np = v2_pred.reshape(-1)
    scalars_gt_np = scalars_gt.reshape(-1)
    scalars_pred_np = scalars_pred.reshape(-1)
    v1_weight_np = v1_weight.reshape(-1)
    v2_weight_np = v2_weight.reshape(-1)

    row_index = epid_gt_np.shape[0] // 2
    gamma_map, gamma_pass = compute_gamma(epid_gt_np, epid_pred_np)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    fig.suptitle(title, fontsize=12)

    def imshow(ax, data, ttl, cmap="inferno", norm=None):
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(ttl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    imshow(axes[0, 0], epid_gt_np, "Actual frame")
    imshow(axes[0, 1], epid_pred_np, "Predicted frame")

    if gamma_map is not None:
        ttl = f"Gamma pass {gamma_pass:.2f}%" if gamma_pass is not None else "Gamma map"
        imshow(axes[0, 2], gamma_map, ttl, cmap=CMAP, norm=NORM)
    else:
        axes[0, 2].text(0.5, 0.5, "Gamma unavailable", ha="center", va="center")
        axes[0, 2].set_axis_off()

    axes[0, 3].plot(epid_gt_np[row_index, :], label="target", color="tab:blue")
    axes[0, 3].plot(epid_pred_np[row_index, :], label="pred", color="tab:red", alpha=0.8)
    axes[0, 3].set_title(f"Line profile (row {row_index})")
    axes[0, 3].legend()

    def plot_leaf_panel(ax, v1_t, v1_p, v2_t, v2_p, jaw_low_t, jaw_low_p, jaw_up_t, jaw_up_p, ttl):
        x_vals = np.linspace(-130, 130, len(v1_t) + 1)
        v1_t_ext = np.append(v1_t, v1_t[-1])
        v1_p_ext = np.append(v1_p, v1_p[-1])
        v2_t_ext = np.append(v2_t, v2_t[-1])
        v2_p_ext = np.append(v2_p, v2_p[-1])
        ax.step(v1_t_ext, -x_vals, where="post", label="v1 target", color="tab:blue")
        ax.step(v1_p_ext, -x_vals, where="post", label="v1 pred", color="tab:blue", linestyle="--")
        ax.step(v2_t_ext, -x_vals, where="post", label="v2 target", color="tab:green")
        ax.step(v2_p_ext, -x_vals, where="post", label="v2 pred", color="tab:green", linestyle="--")
        x_coords = np.linspace(-130, 130, 200)
        ax.plot(x_coords, -np.full_like(x_coords, jaw_low_t), color="tab:blue", alpha=0.8)
        ax.plot(x_coords, -np.full_like(x_coords, jaw_up_t), color="tab:blue", alpha=0.8)
        ax.plot(x_coords, -np.full_like(x_coords, jaw_low_p), color="tab:red", linestyle="--", alpha=0.8)
        ax.plot(x_coords, -np.full_like(x_coords, jaw_up_p), color="tab:red", linestyle="--", alpha=0.8)
        ax.set_title(ttl)
        ax.legend()

    v1_prev_t = v1_gt_np[:52]
    v1_curr_t = v1_gt_np[52:]
    v2_prev_t = v2_gt_np[:52]
    v2_curr_t = v2_gt_np[52:]
    v1_prev_p = v1_pred_np[:52]
    v1_curr_p = v1_pred_np[52:]
    v2_prev_p = v2_pred_np[:52]
    v2_curr_p = v2_pred_np[52:]

    jaw_low_prev_t, jaw_low_curr_t = scalars_gt_np[1], scalars_gt_np[2]
    jaw_up_prev_t, jaw_up_curr_t = scalars_gt_np[3], scalars_gt_np[4]
    jaw_low_prev_p, jaw_low_curr_p = scalars_pred_np[1], scalars_pred_np[2]
    jaw_up_prev_p, jaw_up_curr_p = scalars_pred_np[3], scalars_pred_np[4]

    plot_leaf_panel(
        axes[1, 0],
        v1_prev_t,
        v1_prev_p,
        v2_prev_t,
        v2_prev_p,
        jaw_low_prev_t,
        jaw_low_prev_p,
        jaw_up_prev_t,
        jaw_up_prev_p,
        "Prev CP v1/v2 + jaws",
    )
    plot_leaf_panel(
        axes[1, 1],
        v1_curr_t,
        v1_curr_p,
        v2_curr_t,
        v2_curr_p,
        jaw_low_curr_t,
        jaw_low_curr_p,
        jaw_up_curr_t,
        jaw_up_curr_p,
        "Curr CP v1/v2 + jaws",
    )

    w1_curr = v1_weight_np[52:]
    w2_curr = v2_weight_np[52:]
    x_vals_w = np.linspace(-130, 130, len(w1_curr) + 1)
    w1_ext = np.append(w1_curr, w1_curr[-1])
    w2_ext = np.append(w2_curr, w2_curr[-1])
    axes[1, 2].step(w1_ext, -x_vals_w, where="post", label="v1 weight", color="tab:blue")
    axes[1, 2].step(w2_ext, -x_vals_w, where="post", label="v2 weight", color="tab:green")
    axes[1, 2].set_title("Weights (current CP)")
    axes[1, 2].legend()

    mu_actual = scalars_gt_np[0]
    mu_pred = scalars_pred_np[0]
    bar_width = 0.35
    axes[1, 3].bar(0, mu_actual, bar_width, label="MU actual", color="tab:blue")
    axes[1, 3].bar(bar_width + 0.05, mu_pred, bar_width, label="MU pred", color="tab:red")
    axes[1, 3].set_xticks([])
    axes[1, 3].set_title("MU comparison")
    axes[1, 3].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Visualization saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference + visualization for EPID v5")
    parser.add_argument("--dataset-range", type=str, default="0:1", help="start:end[:step] of datasets")
    parser.add_argument("--dataset_range", type=str, dest="dataset_range", help="Alias for --dataset-range")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        dest="samples_per_dataset",
        default=2,
        help="Random samples per dataset",
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        dest="samples_per_dataset",
        help="Alias for --samples-per-dataset",
    )
    parser.add_argument("--max-total-samples", type=int, default=None, help="Optional global cap")
    parser.add_argument("--checkpoint", type=str, default="Cross_CP/EPID_v5_17Dec25_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--regenerate-missing",
        action="store_true",
        dest="regenerate_missing",
        help="Regenerate dataset if not found (needs KM)",
    )
    parser.add_argument("--output-dir", type=str, default="viz_outputs_v5", help="Directory for PNGs")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset_indices = list(parse_range(args.dataset_range))
    if not dataset_indices:
        raise RuntimeError("No dataset indices parsed")
    print(f"[INFO] Using dataset indices: {dataset_indices}")

    KM = None
    if args.regenerate_missing:
        km_path = Path("data/KM_1500.mat")
        if not km_path.exists():
            raise FileNotFoundError("KM_1500.mat required for regeneration")
        import scipy.io

        KM = scipy.io.loadmat(km_path)["KM_1500"]
        print("[INFO] KM kernel loaded for regeneration")

    datasets = {idx: load_or_generate_dataset(idx, args.regenerate_missing, KM) for idx in dataset_indices}
    for idx, ds in datasets.items():
        print(f"[INFO] Dataset {idx} loaded with {len(ds)} samples")

    encoderunet = EncoderUNet(vector_dim=104, scalar_count=5).to(device)
    unetdecoder = UNetDecoder(vector_dim=104, scalar_count=5).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    enc_state = strip_module_prefix(state.get("encoderunet_state_dict", state))
    dec_state = strip_module_prefix(state.get("unetdecoder_state_dict", state))
    encoderunet.load_state_dict(enc_state, strict=False)
    unetdecoder.load_state_dict(dec_state, strict=False)
    encoderunet.eval()
    unetdecoder.eval()
    print(f"[INFO] Loaded checkpoint {ckpt_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses = [float(x) for x in state.get("train_losses", [])]
    val_losses = [float(x) for x in state.get("val_losses", [])]
    if train_losses or val_losses:
        plot_losses(train_losses, val_losses, out_dir / "losses.png")

    total_drawn = 0
    for ds_idx, ds in datasets.items():
        n = len(ds)
        remaining = args.max_total_samples - total_drawn if args.max_total_samples is not None else None
        per_dataset_cap = args.samples_per_dataset
        if remaining is not None:
            if remaining <= 0:
                print(f"[INFO] Reached max total samples ({args.max_total_samples}); stopping.")
                break
            per_dataset_cap = min(per_dataset_cap, remaining)

        k = min(per_dataset_cap, n)
        chosen = random.sample(range(n), k) if k > 0 else []
        print(f"[INFO] Sampling {k} items from dataset {ds_idx}: indices {chosen}")
        total_drawn += k

        for si in chosen:
            v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p = ds[si]

            v1 = v1.unsqueeze(0).to(device)
            v2 = v2.unsqueeze(0).to(device)
            scalars = scalars.unsqueeze(0).to(device)
            if scalars.dim() == 4 and scalars.size(-1) == 1:
                scalars = scalars.squeeze(-1)

            arrays = ensure_nchw_epid(arrays).to(device)
            arrays_p = ensure_nchw_epid(arrays_p).to(device)

            with torch.no_grad():
                epid_pred, _ = encoderunet(v1, v2, scalars)
                arrays_con = torch.cat([arrays_p, arrays], dim=1)
                v1_pred, v2_pred, scalars_pred, _ = unetdecoder(arrays_con)

            epid_gt = arrays.squeeze(0).cpu().numpy()
            epid_img = epid_pred.squeeze(0).cpu().numpy()
            v1_gt = v1.squeeze(0).cpu().numpy()
            v2_gt = v2.squeeze(0).cpu().numpy()
            v1_pred_np = v1_pred.cpu().numpy()
            v2_pred_np = v2_pred.cpu().numpy()
            scalars_pred_np = scalars_pred.cpu().numpy()
            scalars_gt_np = scalars.squeeze(0).cpu().numpy()
            v1_weight_np = v1_weight.cpu().numpy()
            v2_weight_np = v2_weight.cpu().numpy()

            out_path = out_dir / f"dataset{ds_idx}_sample{si}.png"
            title = f"EPID v5 | Dataset {ds_idx} | Sample {si}"
            visualize(
                epid_gt,
                epid_img,
                v1_gt,
                v2_gt,
                v1_pred_np,
                v2_pred_np,
                scalars_gt_np,
                scalars_pred_np,
                v1_weight_np,
                v2_weight_np,
                out_path,
                title,
            )


if __name__ == "__main__":
    main()
