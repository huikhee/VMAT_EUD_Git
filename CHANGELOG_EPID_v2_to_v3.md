# EPID v2 → v3 changelog

Date: 2025-12-16

This changelog summarizes the functional changes made when creating `generate_and_train_EPID_v3.py` from `generate_and_train_EPID_v2.py`.

## Training / loss changes

- Forward loss weighting is now numerically safer and broadcasts correctly:
  - MU is clamped (`mu = scalars[:,0,0].clamp(min=1e-6)`) to avoid divide-by-zero.
  - The weight map is applied as a true NCHW broadcast (`(B,1,1,1)`) instead of expanding to `(B,H,W)`.
- Leaf ordering constraint (hinge) is now scale-stable:
  - Replaced the summed hinge with a mean squared hinge: `relu(v1_detach - v2)^2.mean()`.
  - `v1` is detached in the hinge so the constraint only pushes `v2` upward.
- Scalar normalization is per-feature instead of a single global scale:
  - Uses `[40,130,130,130,130]` (MU vs jaw-related scalars) so MU error is not under-weighted.

## Model / decoder changes

- Decoder attention resolution increased:
  - Transformer patch size changed from `patch = 8` (64 tokens) to `patch = 4` (256 tokens).

## Optimizer / scheduler changes

- Cosine minimum learning rate lowered:
  - `eta_min` changed from `1e-4` to `1e-6` so annealing actually decays below the base LR.

## Robustness / data shape handling

- EPID tensors are coerced to NCHW before use:
  - Added `ensure_nchw_epid()` and applied it to `arrays`/`arrays_p` in both training and validation.
  - `UNetDecoder.forward()` is hardened to accept inputs shaped `(B,2,1,H,W)` by squeezing the singleton dimension.

## Checkpoint naming

- v3 checkpoints are isolated from v2:
  - Loads/saves `Cross_CP/EPID_v3_16Dec25_checkpoint.pth` (instead of the v2 path) to avoid overwriting.
