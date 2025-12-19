# EPID v4 → v5 changelog

Date: 2025-12-17

This changelog summarizes the functional changes made when creating `generate_and_train_EPID_v5.py` from `generate_and_train_EPID_v4.py`.

## Model / backward-tokenization changes (primary v5 change)

- Backward-path parameter reconstruction (the `UNetDecoder` transformer head) was reworked to use **structured, typed queries** and to decode leaves as **(prev,curr) pairs**.
- Output shapes are intentionally unchanged to preserve the existing training loop and losses:
  - `reconstructed_vector1`: `(B, 104)`
  - `reconstructed_vector2`: `(B, 104)`
  - `reconstructed_scalars`: `(B, 5)`

### v4 backward queries (for reference)

- Learned query tokens were flat and untyped:
  - `v1_query`: 104 tokens → each token outputs 1 scalar → `(B,104)`
  - `v2_query`: 104 tokens → each token outputs 1 scalar → `(B,104)`
  - `s_query`: 5 tokens → each token outputs 1 scalar → `(B,5)`

### v5 backward queries (new)

- Leaves are decoded as **52 tokens per bank**, each predicting a 2-vector `(prev, curr)`:
  - A shared `leaf_query_base` provides 52 base tokens.
  - A learned `bank_embed` distinguishes v1 vs v2 using the same base queries.
  - Leaf outputs use 2D heads:
    - `v1_out2: Linear(d_model → 2)`
    - `v2_out2: Linear(d_model → 2)`
  - The `(prev,curr)` outputs are re-packed to match the training target layout:
    - `[prev52, curr52] → 104`

- Scalars are decoded using **typed queries**:
  - `mu_query`: 1 token → `mu_out: Linear(d_model → 1)` → MU
  - `jaw_query`: 2 tokens (jaw1, jaw2) → `jaw_out: Linear(d_model → 2)` → each token predicts `(prev,curr)`
  - Jaw outputs are flattened to preserve the scalar order:
    - `[jaw1_prev, jaw1_curr, jaw2_prev, jaw2_curr]`
  - Final scalar vector remains:
    - `[MU, jaw1_prev, jaw1_curr, jaw2_prev, jaw2_curr]` (5 values)

## Model / forward-tokenization changes

- No intended functional changes relative to v4.
  - v4 forward tokenization (CLS + 108 tokens) remains the baseline in v5.

## Training / loss changes

- No intended functional changes relative to v4.
  - Loss definitions and normalization (MU clamp + broadcast weights, per-scalar scaling, mean-squared hinge penalty, NCHW coercion) remain the baseline.

## Checkpoint naming

- v5 checkpoints are isolated from v4:
  - Loads/saves `Cross_CP/EPID_v5_17Dec25_checkpoint.pth` (instead of the v4 path) to avoid overwriting.

## Tooling

- Added a matching v5 inference script:
  - `inference_visualize_EPID_v5.py` (imports v5 models, defaults to v5 checkpoint/output folder).
