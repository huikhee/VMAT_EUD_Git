# EPID v5 → v6 changelog

Date: 2025-12-17

This changelog summarizes the functional changes made when creating `generate_and_train_EPID_v6.py` from `generate_and_train_EPID_v5.py`.

## Model / backward-path changes (primary v6 changes)

v6 extends the backward path (`UNetDecoder`) in two ways:

### 1) Separate prev/curr EPID encoders with late fusion

- Instead of feeding a 2-channel input directly into a single encoder, v6 splits the input into:
  - `x_prev = x[:,0:1,:,:]`
  - `x_curr = x[:,1:2,:,:]`
- Each frame is encoded with its own UNet encoder tower (not weight-shared):
  - `encoder{1,2,3}_prev` and `encoder{1,2,3}_curr`.
- Fusion is done late via concatenation + 1×1 conv:
  - fused skip features `f1/f2/f3` and fused bottleneck input `p3`.
- The decoder path is unchanged and uses the fused skip features.

### 2) Multi-scale transformer memory tokens

- Instead of a single 64×64 tokenization, v6 builds transformer memory tokens at 3 scales from `latent_dec`:
  - 64×64 with `patch=4` → 256 tokens
  - 32×32 (avgpool×2) with `patch=4` → 64 tokens
  - 16×16 (avgpool×4) with `patch=4` → 16 tokens
- Tokens are concatenated into one memory sequence (336 tokens total).
- Each scale has its own learned positional embedding and a learned scale embedding:
  - `img_pos_embed_{64,32,16}`
  - `img_scale_embed[3]`

## Backward query head (v5 baseline)

- v5’s structured query head is retained:
  - Leaves decoded as 52 (prev,curr) tokens per bank and repacked to 104.
  - Typed scalar queries: MU (1 token) + jaws (2 tokens) → 5 scalars.

## Forward path

- No intended functional changes relative to v5.
  - v4 forward tokenization (CLS + 108 tokens) remains the baseline.

## Training / loss

- No intended functional changes relative to v5.

## Checkpoint naming

- v6 checkpoints are isolated from v5:
  - Loads/saves `Cross_CP/EPID_v6_17Dec25_checkpoint.pth`.

## Tooling

- Added a matching v6 inference script:
  - `inference_visualize_EPID_v6.py` (imports v6 models, defaults to v6 checkpoint/output folder).
