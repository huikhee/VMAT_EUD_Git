# EPID v7 3 v8 changelog

Date: 2026-01-05

This changelog summarizes the functional and training-pipeline changes made when creating and iterating on `generate_and_train_EPID_v8.py` from `generate_and_train_EPID_v7.py`.

## Distributed training / dataset assignment (primary v8 change)

### 1) Randomized dataset0rank assignment (deterministic)

- v7 assigned contiguous blocks of dataset IDs per rank (`rank * (640/world_size) 3 ...`).
- v8 assigns dataset IDs by:
  1) building a list of eligible dataset IDs
  2) (optionally) shuffling that list deterministically using a seed
  3) assigning by striding: `assigned_dataset_ids = dataset_ids[rank::world_size]`

Why this matters:
- Avoids bias from any sequential pattern in dataset IDs (e.g. regular 3 semi-regular 3 irregular ordering).
- Keeps work balanced across ranks even after shuffling.

### 2) Optional dataset subset selection for smoke tests

v8 adds environment-variable controls to train on a subset of dataset IDs without editing code.

- `EPID_DATASET_RANGE` supports a Python-style range string `start:stop:step` (example: `0:640:80`).
- `EPID_DATASET_IDS` supports an explicit comma-separated list (example: `0,80,160`).

## Training pipeline changes

### 1) Timing log output is minimal by default

- v8 reduces timing log verbosity by default to minimize console spam.
- Detailed timing can be re-enabled via environment variables (see Variables section).

### 2) Batch size + world size defaults (match v7 behavior)

- v7 used a per-rank batch size of `128 // world_size`.
- v8 now matches v7 by default:
  - `batch_size = 128 // world_size`
  - default `world_size = 1` (unless overridden)

Notes:
- Earlier v8 iterations used different defaults (e.g., `world_size=2` and/or fixed per-rank batch), which changes the global batch size and number of optimizer steps per epoch and can change convergence.

### 3) DataLoader tuning remains environment-controlled

v8 keeps the v7 pattern of cluster-safe DataLoader defaults with env overrides.

## DDP rendezvous

- v8 uses a different default `MASTER_PORT` constant than v7.
- If you run multiple distributed jobs on the same node, port collisions can still occur.
  - Prefer setting `MASTER_PORT` externally (job scheduler) or using `torchrun`-style launch.

## Tooling

- Added `inference_visualize_EPID_v8.py` as the v8 counterpart to v7 visualization/inference.
- Extended checkpoint comparison tooling to include v7 and v8:
  - `compare_checkpoints_epid_v3_v8.py` (renamed from the earlier v33v6 script).

## Variables and how to run

### Environment variables used by `generate_and_train_EPID_v8.py`

Dataset assignment / selection:
- `EPID_DATASET_SHUFFLE` (default `1`): set to `0` to disable dataset ID shuffling.
- `EPID_DATASET_SEED` (default `12345`): seed used for deterministic dataset shuffling.
- `EPID_DATASET_RANGE` (default empty): restrict dataset IDs via `start:stop:step`.
- `EPID_DATASET_IDS` (default empty): restrict dataset IDs via `id1,id2,id3`.

Distributed / batch:
- `EPID_WORLD_SIZE` (default `1`): number of spawned DDP processes.
- `EPID_BATCH_SIZE` (default empty): overrides per-rank batch size. If unset, uses `128 // world_size`.

DataLoader:
- `EPID_NUM_WORKERS` (default `0`)
- `EPID_PIN_MEMORY` (default `1`)
- `EPID_PREFETCH_FACTOR` (default `2`, only used when `EPID_NUM_WORKERS > 0`)

Timing:
- `EPID_TIMING_EVERY` (default `0`): `0` means minimal timing output; 31 prints every N batches.
- `EPID_TIMING_MAX` (default `1`): maximum timing lines printed per epoch.

### Run (no environment variables)

PowerShell:

```powershell
cd C:\Users\MRP_U\torch_data\VMAT_EUD_Git
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

This uses:
- `world_size = 1`
- `batch_size = 128`
- full dataset ID list `0..639`
- deterministic shuffle ON by default

### Run with environment variables (PowerShell examples)

1) Match v7-style behavior as closely as possible (single process, no dataset shuffle):

```powershell
$env:EPID_WORLD_SIZE = "1"
$env:EPID_DATASET_SHUFFLE = "0"
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

2) Two GPUs (two ranks), v7-style per-rank batch scaling (global batch stays ~128):

```powershell
$env:EPID_WORLD_SIZE = "2"
# EPID_BATCH_SIZE not set => per-rank batch = 128//2 = 64
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

3) Two GPUs, fixed per-rank batch size (global batch increases):

```powershell
$env:EPID_WORLD_SIZE = "2"
$env:EPID_BATCH_SIZE = "128"   # per-rank
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

4) Quick smoke test on a subset of datasets:

```powershell
$env:EPID_WORLD_SIZE = "1"
$env:EPID_DATASET_RANGE = "0:640:80"  # 0,80,160,...
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

5) Enable more timing output:

```powershell
$env:EPID_TIMING_EVERY = "50"
$env:EPID_TIMING_MAX = "20"
C:\Users\MRP_U\miniconda3\envs\torch\python.exe .\generate_and_train_EPID_v8.py
```

### Clear environment variables (PowerShell)

```powershell
Remove-Item Env:EPID_WORLD_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:EPID_BATCH_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:EPID_DATASET_RANGE -ErrorAction SilentlyContinue
Remove-Item Env:EPID_DATASET_IDS -ErrorAction SilentlyContinue
Remove-Item Env:EPID_DATASET_SHUFFLE -ErrorAction SilentlyContinue
Remove-Item Env:EPID_DATASET_SEED -ErrorAction SilentlyContinue
Remove-Item Env:EPID_NUM_WORKERS -ErrorAction SilentlyContinue
Remove-Item Env:EPID_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:EPID_PREFETCH_FACTOR -ErrorAction SilentlyContinue
Remove-Item Env:EPID_TIMING_EVERY -ErrorAction SilentlyContinue
Remove-Item Env:EPID_TIMING_MAX -ErrorAction SilentlyContinue
```
