# EPID v6 → v7 changelog

Date: 2025-12-23

This changelog summarizes the functional and training-pipeline changes made when creating and iterating on `generate_and_train_EPID_v7.py` from `generate_and_train_EPID_v6.py`.

## Data / geometry (primary v7 change)

### 1) EPID grid is now native 256×256 at 1 mm pitch

- EPID masks and KM-convolved EPID images are generated and trained on a 256×256 grid.
- The intended physical field-of-view is centered and represented as coordinates in $[-128, 128)$ mm.
- Data folder for v7 artifacts is separated:
  - Datasets are saved/loaded from `VMAT_Art_data_256/`.

### 2) Dataset sample count semantics clarified

- Each saved dataset file contains 2050 control points/frames.
- The dataset length is `2050 - 2 = 2048` samples, matching the prev/curr pairing logic.
- `CustomDataset.__getitem__` yields paired `(prev,curr)` features as before.

## Training pipeline / performance (major operational improvements)

### 1) Remove repeated DataLoader startup overhead

- v6/v7 originally used “one DataLoader per dataset id” (hundreds of loaders per epoch).
- v7 now builds a single `ConcatDataset` per split (train/val) and uses a single DataLoader per split.
- This change dramatically reduces `data_wait` stalls on fast GPUs (e.g., H100) by avoiding repeated iterator startup/first-batch costs.

### 2) DataLoader knobs are configurable via environment variables

- `EPID_NUM_WORKERS` (default `0`)
- `EPID_PIN_MEMORY` (default `1`)
- `EPID_PREFETCH_FACTOR` (default `2`, only applied when `EPID_NUM_WORKERS > 0`)

### 3) Non-blocking host→device transfers

- `.to(device, non_blocking=True)` is used for batch tensors when `pin_memory=True`, enabling overlap where possible.

### 4) Per-batch timing instrumentation for bottleneck diagnosis

- v7 prints rate-limited per-batch timing lines to diagnose GPU stalls:
  - `[TIMING][train] ... data_wait=... h2d=... gpu_compute=... iter=... other=...`
  - `[TIMING][val] ...`
- Controlled by:
  - `EPID_TIMING_EVERY` (default `50`)
  - `EPID_TIMING_MAX` (default `20`)

### 5) `CustomDataset` batch-prep CPU optimization (safe + backwards compatible)

- `CustomDataset` now precomputes prev/curr paired tensors (vectors/weights/scalars) to reduce per-sample Python overhead.
- Large EPID arrays are *not duplicated* in RAM (uses slicing views rather than fancy indexing).
- Backwards compatibility: datasets loaded via `torch.load()` may not have new cached attributes (because pickle restore does not call `__init__`).
  - v7 uses a lazy `_ensure_pairs()` path to construct caches on first use.

### 6) Safety warning for batch divisibility

- A one-time warning is emitted if the per-dataset train/val split lengths are not divisible by the default batch size, since within-batch “consistency” losses assume consecutive samples belong to the same underlying sequence.

## Model architecture

- No intended architectural changes vs v6 in the forward/backward networks.
- The primary v7 work is the data geometry and the training/data pipeline to support 256×256 efficiently.

## Checkpoint naming

- v7 checkpoints are isolated from v6:
  - Saves `Cross_CP/EPID_v7_20Dec25_checkpoint.pth`.

## Tooling

- v7 has a matching visualization/inference script:
  - `inference_visualize_EPID_v7.py` (updated to the 256×256 geometry).
