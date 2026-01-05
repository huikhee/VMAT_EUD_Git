"""Compare EPID checkpoints (v3-v8) on a shared evaluation subset.

This script evaluates:
- Forward path: params -> EPID (MSE/MAE)
- Backward path: EPID(prev,curr) -> params (leaf/scalar MAE; optional weighted leaf MAE)
- Optional gamma pass rate via pymedphys (capped for speed)

It also prints checkpoint loss-history summaries (min/last val).

Usage example:
    python compare_checkpoints_epid_v3_v6.py --dataset-range 0:640:80 --samples-per-dataset 5 --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import generate_and_train_EPID_v3 as v3
import generate_and_train_EPID_v4 as v4
import generate_and_train_EPID_v5 as v5
import generate_and_train_EPID_v6 as v6
import generate_and_train_EPID_v7 as v7
import generate_and_train_EPID_v8 as v8


# NOTE:
# Some of the saved dataset .pt files were created by running the generator as a script,
# so the pickled class path is "__main__.CustomDataset".
# When loading those files from this script, Python's __main__ is *this* module.
# Defining CustomDataset here allows torch.load() to unpickle those datasets.
class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        if hasattr(self, "vector1"):
            n = int(self.vector1.size(0))
            # v3 stores sequential CPs and builds prev/curr pairs in __getitem__.
            if not hasattr(self, "arrays_p"):
                return max(0, n - 2)
            return n
        if hasattr(self, "arrays"):
            return int(self.arrays.size(0))
        raise AttributeError("Unrecognized dataset object: missing vector1/arrays")

    @staticmethod
    def _get_scalars_row(obj, idx: int) -> torch.Tensor:
        """Extract a single (1,5) scalars row across dataset versions.

        Supports:
        - scalar1 stored as (N,5)
        - scalar1..scalar5 stored separately
        - legacy v3-style (scalar1 + scalar2/3 prev/curr)
        """
        if hasattr(obj, "scalar1"):
            s1 = getattr(obj, "scalar1")
            if torch.is_tensor(s1) and s1.dim() == 2 and s1.size(1) == 5:
                return s1[idx].unsqueeze(0)

        if all(hasattr(obj, f"scalar{i}") for i in range(1, 6)):
            vals = []
            for i in range(1, 6):
                si = getattr(obj, f"scalar{i}")[idx]
                if torch.is_tensor(si) and si.dim() > 0:
                    si = si.reshape(-1)[0]
                vals.append(si)
            return torch.stack(vals, dim=0).unsqueeze(0)

        prev_idx = idx - 1
        scalar1 = obj.scalar1[idx].reshape(-1)[0].unsqueeze(0).unsqueeze(0)
        scalar2_prev = obj.scalar2[prev_idx].reshape(-1)[0].unsqueeze(0).unsqueeze(0)
        scalar2_cur = obj.scalar2[idx].reshape(-1)[0].unsqueeze(0).unsqueeze(0)
        scalar3_prev = obj.scalar3[prev_idx].reshape(-1)[0].unsqueeze(0).unsqueeze(0)
        scalar3_cur = obj.scalar3[idx].reshape(-1)[0].unsqueeze(0).unsqueeze(0)
        return torch.cat([scalar1, scalar2_prev, scalar2_cur, scalar3_prev, scalar3_cur], dim=1)

    def __getitem__(self, idx):
        # v4-v6 datasets typically store explicit prev/current EPID frames.
        if hasattr(self, "arrays_p"):
            v1 = self.vector1[idx].unsqueeze(0)
            v2 = self.vector2[idx].unsqueeze(0)

            # Scalars may be stored as 5 separate arrays (scalar1..scalar5) in newer versions.
            if all(hasattr(self, f"scalar{i}") for i in range(1, 6)):
                s = [getattr(self, f"scalar{i}")[idx].unsqueeze(0).unsqueeze(0) for i in range(1, 6)]
                scalars = torch.cat(s, dim=1)
            else:
                # Fallback: treat scalar1 as (N,5) tensor if present.
                scalars = self.scalar1[idx].unsqueeze(0)
                if scalars.dim() == 1:
                    scalars = scalars.unsqueeze(0)

            v1_weight = getattr(self, "v1_weight", getattr(self, "vector1_weight", torch.ones_like(v1)))
            v2_weight = getattr(self, "v2_weight", getattr(self, "vector2_weight", torch.ones_like(v2)))
            if v1_weight.dim() == 2:
                v1_weight = v1_weight[idx].unsqueeze(0)
            if v2_weight.dim() == 2:
                v2_weight = v2_weight[idx].unsqueeze(0)

            arrays = self.arrays[idx].unsqueeze(0)
            arrays_p = self.arrays_p[idx].unsqueeze(0)
            return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p

        # v3-style: build prev/curr pairing on the fly
        idx = int(idx) + 2
        prev_idx = idx - 1

        v1 = torch.cat([self.vector1[prev_idx].unsqueeze(0), self.vector1[idx].unsqueeze(0)], dim=1)
        v2 = torch.cat([self.vector2[prev_idx].unsqueeze(0), self.vector2[idx].unsqueeze(0)], dim=1)

        w1 = getattr(self, "vector1_weight", torch.ones_like(self.vector1))
        w2 = getattr(self, "vector2_weight", torch.ones_like(self.vector2))
        v1_weight = torch.cat([w1[prev_idx].unsqueeze(0), w1[idx].unsqueeze(0)], dim=1)
        v2_weight = torch.cat([w2[prev_idx].unsqueeze(0), w2[idx].unsqueeze(0)], dim=1)

        scalars = self._get_scalars_row(self, idx)

        arrays = self.arrays[idx].unsqueeze(0)
        arrays_p = self.arrays[prev_idx].unsqueeze(0)
        return v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p


def parse_range(spec: str) -> Iterable[int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("range must be start:end or start:end:step")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    return range(start, end, step)


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


def summarize_losses(arr: List[float] | list) -> Dict[str, float | int]:
    arr = list(arr) if arr is not None else []
    if not arr:
        return {"n": 0}
    arr_f = [float(x) for x in arr]
    m = min(arr_f)
    return {"n": len(arr_f), "last": float(arr_f[-1]), "min": float(m), "argmin": int(arr_f.index(m) + 1)}


def _safe_float_list(x) -> List[float]:
    if x is None:
        return []
    return [float(v) for v in list(x)]


def history_only_metrics(state: dict, tail_n: int = 25) -> Dict[str, float | int]:
    """Compute comparison metrics using only loss histories stored in a checkpoint."""

    train = _safe_float_list(state.get("train_losses", []))
    val = _safe_float_list(state.get("val_losses", []))
    n = min(len(train), len(val)) if (train and val) else max(len(train), len(val))

    out: Dict[str, float | int] = {"epochs": int(n)}
    if train:
        out["train_last"] = float(train[-1])
        out["train_min"] = float(min(train))
        out["train_mean"] = float(np.mean(train))
    if val:
        out["val_last"] = float(val[-1])
        out["val_min"] = float(min(val))
        out["val_argmin"] = int(val.index(min(val)) + 1)
        out["val_mean"] = float(np.mean(val))

    if train and val:
        out["gap_last"] = float(val[-1] - train[-1])
        # AUC (integral) is length-dependent; mean is scale-stable. Keep both.
        out["train_auc"] = float(np.trapz(train, dx=1.0))
        out["val_auc"] = float(np.trapz(val, dx=1.0))

        # Tail slope: linear fit on last N epochs (negative is better/faster convergence)
        t = min(tail_n, len(val))
        if t >= 2:
            y = np.asarray(val[-t:], dtype=float)
            x = np.arange(t, dtype=float)
            slope = float(np.polyfit(x, y, deg=1)[0])
            out["val_tail_slope"] = slope

    return out


def print_history_only_table(ckpt_states: Dict[str, dict], tail_n: int) -> None:
    rows: List[Tuple[str, Dict[str, float | int]]] = []
    val_series: Dict[str, List[float]] = {}
    train_series: Dict[str, List[float]] = {}

    for ver, state in ckpt_states.items():
        m = history_only_metrics(state, tail_n=tail_n)
        rows.append((ver, m))
        val_series[ver] = _safe_float_list(state.get("val_losses", []))
        train_series[ver] = _safe_float_list(state.get("train_losses", []))

    # Align comparisons to the same epoch window when possible.
    common_epochs = None
    epoch_counts = [int(m.get("epochs", 0)) for _, m in rows if int(m.get("epochs", 0)) > 0]
    if epoch_counts:
        common_epochs = min(epoch_counts)

    def g(m, k, default="-"):
        return m.get(k, default)

    print("\n=== History-only comparison (from train_losses/val_losses) ===")
    print(
        "ver  epochs  train_last  val_last  val_min@ep   gap_last  val_mean  val_mean_common  val_mean_lastN  val_tail_slope"
    )
    for ver, m in rows:
        epochs = g(m, "epochs", 0)
        train_last = g(m, "train_last")
        val_last = g(m, "val_last")
        val_min = g(m, "val_min")
        val_argmin = g(m, "val_argmin")
        gap_last = g(m, "gap_last")
        val_mean = g(m, "val_mean")
        slope = g(m, "val_tail_slope")

        # Common-window mean (first common_epochs epochs)
        val_mean_common = "-"
        if common_epochs is not None:
            vs = val_series.get(ver, [])
            if len(vs) >= 1:
                val_mean_common = float(np.mean(vs[:common_epochs]))

        # Last-N mean
        val_mean_lastn = "-"
        vs2 = val_series.get(ver, [])
        if vs2:
            t = min(tail_n, len(vs2))
            val_mean_lastn = float(np.mean(vs2[-t:]))

        def f6(x):
            return f"{float(x):.6f}" if x != "-" else "-"

        def f4(x):
            return f"{float(x):.4f}" if x != "-" else "-"

        def f4sci(x):
            return f"{float(x):.3e}" if x != "-" else "-"

        val_min_ep = "-"
        if val_min != "-" and val_argmin != "-":
            val_min_ep = f"{float(val_min):.6f}@{int(val_argmin)}"

        print(
            f"{ver:>2}  {int(epochs):>5}  {f6(train_last):>9}  {f6(val_last):>7}  {val_min_ep:>10}  {f6(gap_last):>8}  {f6(val_mean):>8}"
            f"  {f6(val_mean_common):>13}  {f6(val_mean_lastn):>12}  {f4sci(slope):>13}"
        )

    if common_epochs is not None:
        print(f"[INFO] val_mean_common computed over first {common_epochs} epochs")

    # Make the "end" comparison explicit (lower is better)
    def _rank(key: str) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for ver, m in rows:
            v = m.get(key)
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                out.append((ver, float(v)))
        out.sort(key=lambda t: t[1])
        return out

    train_last_rank = _rank("train_last")
    val_last_rank = _rank("val_last")
    val_min_rank = _rank("val_min")

    if train_last_rank:
        print("[RANK] train_last (lower=better): " + " < ".join([f"{v}({x:.6f})" for v, x in train_last_rank]))
    if val_last_rank:
        print("[RANK] val_last   (lower=better): " + " < ".join([f"{v}({x:.6f})" for v, x in val_last_rank]))
    if val_min_rank:
        print("[RANK] val_min    (lower=better): " + " < ".join([f"{v}({x:.6f})" for v, x in val_min_rank]))


def plot_loss_histories(
    rows: List[Tuple[str, str, Dict[str, float | int], Dict[str, float | int], Metrics]],
    ckpt_states: Dict[str, dict],
    out_path: Path,
    logy: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in your torch env (e.g. conda install matplotlib)."
        ) from e

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for ver, _, _, _, _ in rows:
        state = ckpt_states.get(ver)
        if state is None:
            continue
        train = [float(x) for x in state.get("train_losses", [])]
        val = [float(x) for x in state.get("val_losses", [])]

        if train:
            ax.plot(range(1, len(train) + 1), train, label=f"{ver} train", linewidth=1.5)
        if val:
            ax.plot(range(1, len(val) + 1), val, label=f"{ver} val", linewidth=1.5, linestyle="--")

            vmin = min(val)
            argmin = int(val.index(vmin) + 1)
            ax.scatter([argmin], [vmin], s=18)

    ax.set_title("Training history comparison (v3–v8)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    if logy:
        ax.set_yscale("log")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def try_import_pymedphys() -> Optional[object]:
    try:
        import pymedphys  # type: ignore

        return pymedphys
    except Exception:
        return None


def compute_gamma_pass_rate(pymedphys: object, ref: np.ndarray, eva: np.ndarray) -> Optional[float]:
    ref = np.asarray(ref, dtype=float).squeeze()
    eva = np.asarray(eva, dtype=float).squeeze()
    if ref.ndim != 2 or eva.ndim != 2 or ref.shape != eva.shape:
        return None

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
        return None

    valid = np.isfinite(gamma_map)
    if not np.any(valid):
        return None
    return float(100.0 * np.mean((gamma_map[valid] <= 1.0)))


@dataclass
class Sample:
    v1: torch.Tensor
    v2: torch.Tensor
    scalars: torch.Tensor
    v1_weight: torch.Tensor
    v2_weight: torch.Tensor
    arrays: torch.Tensor
    arrays_p: torch.Tensor


def load_dataset_file(path: Path):
    # Ensure all possible CustomDataset classes are importable for unpickling
    _ = (v3.CustomDataset, v4.CustomDataset, v5.CustomDataset, v6.CustomDataset, v7.CustomDataset, v8.CustomDataset)
    return torch.load(path, map_location="cpu")


def load_samples(
    data_dir: Path,
    dataset_indices: List[int],
    samples_per_dataset: int,
    max_total_samples: Optional[int],
    regenerate_missing: bool,
) -> Tuple[List[Sample], Dict[int, int]]:
    """Load a reproducible subset of samples across datasets."""

    km = None
    if regenerate_missing:
        km_path = Path("data/KM_1500.mat")
        if not km_path.exists():
            raise FileNotFoundError("KM_1500.mat required for regeneration")
        import scipy.io

        km = scipy.io.loadmat(km_path)["KM_1500"]

    samples: List[Sample] = []
    sizes: Dict[int, int] = {}

    total_drawn = 0
    for ds_idx in dataset_indices:
        file_path = data_dir / f"Art_dataset_coll0_{ds_idx}.pt"
        if not file_path.exists():
            if not regenerate_missing:
                raise FileNotFoundError(f"Missing dataset file: {file_path}")
            if km is None:
                raise RuntimeError("KM kernel required")
            # Use v6 generator (latest) to regenerate on demand
            v6.generate_and_save_dataset(ds_idx, km)
            if not file_path.exists():
                raise RuntimeError(f"Regeneration did not create {file_path}")

        ds = load_dataset_file(file_path)
        n = len(ds)
        sizes[ds_idx] = n

        remaining = (max_total_samples - total_drawn) if max_total_samples is not None else None
        per_ds = samples_per_dataset
        if remaining is not None:
            if remaining <= 0:
                break
            per_ds = min(per_ds, remaining)

        k = min(per_ds, n)
        chosen = list(range(n)) if k == n else random.sample(range(n), k)
        total_drawn += k

        for si in chosen:
            v1, v2, scalars, v1_weight, v2_weight, arrays, arrays_p = ds[si]
            samples.append(
                Sample(
                    v1=v1,
                    v2=v2,
                    scalars=scalars,
                    v1_weight=v1_weight,
                    v2_weight=v2_weight,
                    arrays=arrays,
                    arrays_p=arrays_p,
                )
            )

    return samples, sizes


@dataclass
class Metrics:
    n: int
    forward_mse: float
    forward_mae: float
    leaf_mae: float
    leaf_mae_weighted: Optional[float]
    mu_mae: float
    jaws_mae: float
    gamma_pass_mean: Optional[float]


def eval_model(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    samples: List[Sample],
    gamma_pymedphys: Optional[object],
    gamma_max_samples: int,
) -> Metrics:
    encoder.eval()
    decoder.eval()

    mse_sum = 0.0
    mae_sum = 0.0

    leaf_abs_sum = 0.0
    leaf_count = 0

    leaf_abs_w_sum = 0.0
    leaf_w_sum = 0.0

    mu_abs_sum = 0.0
    jaw_abs_sum = 0.0

    gamma_vals: List[float] = []

    # Cap gamma for speed
    gamma_indices = set()
    if gamma_pymedphys is not None and gamma_max_samples > 0:
        gamma_indices = set(range(min(len(samples), gamma_max_samples)))

    with torch.no_grad():
        for idx, s in enumerate(samples):
            v1 = s.v1.unsqueeze(0).to(device)
            v2 = s.v2.unsqueeze(0).to(device)

            scalars = s.scalars.unsqueeze(0).to(device)
            if scalars.dim() == 4 and scalars.size(-1) == 1:
                scalars = scalars.squeeze(-1)

            arrays = ensure_nchw_epid(s.arrays).to(device)
            arrays_p = ensure_nchw_epid(s.arrays_p).to(device)

            epid_pred, _ = encoder(v1, v2, scalars)
            epid_gt = arrays

            if epid_pred.shape != epid_gt.shape:
                epid_pred = F.interpolate(epid_pred, size=epid_gt.shape[-2:], mode="bilinear", align_corners=False)

            diff = epid_pred - epid_gt
            mse_sum += float(torch.mean(diff * diff).item())
            mae_sum += float(torch.mean(torch.abs(diff)).item())

            arrays_con = torch.cat([arrays_p, arrays], dim=1)
            v1_pred, v2_pred, scalars_pred, _ = decoder(arrays_con)

            # Leaf vectors
            v1_gt = v1.squeeze(0)
            v2_gt = v2.squeeze(0)
            v1_pred = v1_pred.squeeze(0)
            v2_pred = v2_pred.squeeze(0)

            leaf_err = torch.cat([v1_pred - v1_gt, v2_pred - v2_gt], dim=0).reshape(-1)
            leaf_abs_sum += float(torch.sum(torch.abs(leaf_err)).item())
            leaf_count += int(leaf_err.numel())

            # Weighted leaf MAE (uses provided weights, duplicated for v1/v2)
            try:
                w1 = s.v1_weight.reshape(-1).to(device)
                w2 = s.v2_weight.reshape(-1).to(device)
                w = torch.cat([w1, w2], dim=0).reshape(-1)
                w = torch.clamp(w, min=0.0)
                leaf_abs_w_sum += float(torch.sum(torch.abs(leaf_err) * w).item())
                leaf_w_sum += float(torch.sum(w).item())
            except Exception:
                pass

            # Scalars
            scalars_gt = scalars.squeeze(0).reshape(-1)
            scalars_pred = scalars_pred.squeeze(0).reshape(-1)
            if scalars_gt.numel() >= 5 and scalars_pred.numel() >= 5:
                mu_abs_sum += float(torch.abs(scalars_pred[0] - scalars_gt[0]).item())
                jaw_abs_sum += float(torch.mean(torch.abs(scalars_pred[1:5] - scalars_gt[1:5])).item())

            # Gamma
            if gamma_pymedphys is not None and idx in gamma_indices:
                g = compute_gamma_pass_rate(gamma_pymedphys, epid_gt.squeeze(0).cpu().numpy(), epid_pred.squeeze(0).cpu().numpy())
                if g is not None:
                    gamma_vals.append(float(g))

    n = len(samples)
    leaf_mae = (leaf_abs_sum / leaf_count) if leaf_count > 0 else float("nan")
    leaf_mae_weighted = None
    if leaf_w_sum > 0:
        leaf_mae_weighted = leaf_abs_w_sum / leaf_w_sum

    gamma_mean = None
    if gamma_vals:
        gamma_mean = float(np.mean(gamma_vals))

    return Metrics(
        n=n,
        forward_mse=mse_sum / n if n else float("nan"),
        forward_mae=mae_sum / n if n else float("nan"),
        leaf_mae=leaf_mae,
        leaf_mae_weighted=leaf_mae_weighted,
        mu_mae=mu_abs_sum / n if n else float("nan"),
        jaws_mae=jaw_abs_sum / n if n else float("nan"),
        gamma_pass_mean=gamma_mean,
    )


def load_models_for_version(version: str, device: torch.device):
    if version == "v3":
        mod = v3
    elif version == "v4":
        mod = v4
    elif version == "v5":
        mod = v5
    elif version == "v6":
        mod = v6
    elif version == "v7":
        mod = v7
    elif version == "v8":
        mod = v8
    else:
        raise ValueError(version)

    enc = mod.EncoderUNet(vector_dim=104, scalar_count=5).to(device)
    dec = mod.UNetDecoder(vector_dim=104, scalar_count=5).to(device)
    return enc, dec


def load_checkpoint_into(enc, dec, ckpt_path: Path) -> Tuple[dict, Dict[str, float | int], Dict[str, float | int]]:
    state = torch.load(ckpt_path, map_location="cpu")
    enc_state = strip_module_prefix(state.get("encoderunet_state_dict", state))
    dec_state = strip_module_prefix(state.get("unetdecoder_state_dict", state))
    enc.load_state_dict(enc_state, strict=False)
    dec.load_state_dict(dec_state, strict=False)

    train = summarize_losses(state.get("train_losses", []))
    val = summarize_losses(state.get("val_losses", []))
    return state, train, val


def fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EPID v3-v8 checkpoints")
    parser.add_argument("--dataset-range", type=str, default="0:640:80")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--max-total-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="VMAT_Art_data",
        help="Folder containing Art_dataset_coll0_*.pt (v7/v8 often use VMAT_Art_data_256).",
    )
    parser.add_argument("--regenerate-missing", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gamma-max-samples", type=int, default=10)
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only compare checkpoints using stored loss histories (skip dataset loading + eval metrics).",
    )
    parser.add_argument(
        "--history-tail-n",
        type=int,
        default=25,
        help="Number of last epochs used to estimate tail convergence slope.",
    )

    parser.add_argument(
        "--loss-plot-path",
        type=str,
        default="loss_history_v3_v8.png",
        help="Where to save the training/validation loss-history plot (PNG).",
    )
    parser.add_argument(
        "--loss-plot-logy",
        action="store_true",
        help="Use log scale for loss plot y-axis.",
    )

    parser.add_argument("--ckpt-v3", type=str, default="Cross_CP/EPID_v3_16Dec25_checkpoint.pth")
    parser.add_argument("--ckpt-v4", type=str, default="Cross_CP/EPID_v4_17Dec25_checkpoint.pth")
    parser.add_argument("--ckpt-v5", type=str, default="Cross_CP/EPID_v5_17Dec25_checkpoint.pth")
    parser.add_argument("--ckpt-v6", type=str, default="Cross_CP/EPID_v6_17Dec25_checkpoint.pth")
    parser.add_argument("--ckpt-v7", type=str, default="Cross_CP/EPID_v7_20Dec25_checkpoint.pth")
    parser.add_argument("--ckpt-v8", type=str, default="Cross_CP/EPID_v8_23Dec25_checkpoint.pth")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    dataset_indices = list(parse_range(args.dataset_range))
    if not args.history_only:
        print(f"[INFO] Datasets: {dataset_indices}")

        samples, sizes = load_samples(
            data_dir=data_dir,
            dataset_indices=dataset_indices,
            samples_per_dataset=args.samples_per_dataset,
            max_total_samples=args.max_total_samples,
            regenerate_missing=args.regenerate_missing,
        )
        print(f"[INFO] Loaded {len(samples)} samples total")
        for k in dataset_indices:
            if k in sizes:
                print(f"  dataset {k}: n={sizes[k]}")

        pymedphys = try_import_pymedphys()
        if pymedphys is None:
            print("[WARN] pymedphys not available; gamma will be skipped")
    else:
        samples = []
        pymedphys = None

    versions = {
        "v3": Path(args.ckpt_v3),
        "v4": Path(args.ckpt_v4),
        "v5": Path(args.ckpt_v5),
        "v6": Path(args.ckpt_v6),
        "v7": Path(args.ckpt_v7),
        "v8": Path(args.ckpt_v8),
    }

    rows = []
    ckpt_states: Dict[str, dict] = {}
    for ver, ckpt in versions.items():
        if not ckpt.exists():
            print(f"[WARN] Missing checkpoint for {ver}: {ckpt}")
            continue

        enc, dec = load_models_for_version(ver, device)
        state, train_s, val_s = load_checkpoint_into(enc, dec, ckpt)
        ckpt_states[ver] = state

        if args.history_only:
            metrics = Metrics(
                n=0,
                forward_mse=float("nan"),
                forward_mae=float("nan"),
                leaf_mae=float("nan"),
                leaf_mae_weighted=None,
                mu_mae=float("nan"),
                jaws_mae=float("nan"),
                gamma_pass_mean=None,
            )
        else:
            metrics = eval_model(
                enc,
                dec,
                device,
                samples,
                gamma_pymedphys=pymedphys,
                gamma_max_samples=args.gamma_max_samples,
            )

        rows.append((ver, ckpt.name, train_s, val_s, metrics))

    # Pretty print
    print("\n=== Loss history (from checkpoints) ===")
    for ver, ckpt_name, train_s, val_s, _ in rows:
        print(
            f"{ver:>2} {ckpt_name} | train min={train_s.get('min','-')} last={train_s.get('last','-')} "
            f"| val min={val_s.get('min','-')} (epoch {val_s.get('argmin','-')}) last={val_s.get('last','-')}"
        )

    print_history_only_table(ckpt_states, tail_n=args.history_tail_n)

    if not args.history_only:
        print("\n=== Eval metrics (same samples for all) ===")
        header = (
            "ver  forward_mse  forward_mae  leaf_mae  leaf_mae_w  mu_mae  jaws_mae  gamma_pass"
        )
        print(header)
        for ver, _, _, _, m in rows:
            print(
                f"{ver:>2}   {m.forward_mse:.6f}    {m.forward_mae:.6f}   {m.leaf_mae:.4f}   {fmt(m.leaf_mae_weighted):>9}  "
                f"{m.mu_mae:.4f}  {m.jaws_mae:.4f}   {fmt(m.gamma_pass_mean):>9}"
            )

    # Plot training history
    out_path = Path(args.loss_plot_path)
    plot_loss_histories(rows, ckpt_states, out_path=out_path, logy=args.loss_plot_logy)
    print(f"\n[INFO] Saved loss-history plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()
