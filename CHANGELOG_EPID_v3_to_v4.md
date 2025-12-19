# EPID v3 → v4 changelog

Date: 2025-12-17

This changelog summarizes the functional changes made when creating `generate_and_train_EPID_v4.py` from `generate_and_train_EPID_v3.py`.

## Model / forward-tokenization changes (primary v4 change)

- Forward-path parameter encoding was reworked to use an explicit CLS pooling token and a larger, reorganized token sequence.

### v3 forward tokenization (for reference)

- Tokens were:
  - 1 “global scalar” token (all 5 scalars embedded together)
  - 52 leaf tokens, each with 4 features: `(v1_prev, v1_curr, v2_prev, v2_curr)`
- Pooling used the transformer output at token index 0 (the scalar token).
- Positional embedding length was `1 + leaf_bins` (i.e., 53 tokens).

### v4 forward tokenization (new)

- Tokens are now **108 total**:
  - 1 **CLS** token (learned; used for pooling)
  - 1 **MU** token
  - 2 **jaw** tokens (each carries `[prev, curr]`)
  - 104 **leaf** tokens (each carries `[prev, curr]`)
    - leaf tokens 1–52 from `v1`: `[v1[i], v1[i+52]]`
    - leaf tokens 53–104 from `v2`: `[v2[i], v2[i+52]]`
- Token embeddings are split by type:
  - `mu_token_embed: Linear(1 → d_model)`
  - `jaw_token_embed: Linear(2 → d_model)`
  - `leaf_token_embed: Linear(2 → d_model)`
  - plus a learned `cls_token: (1,1,d_model)`
- Pooling now uses the CLS token output (token index 0 by construction).
- Positional embedding was resized to match the new sequence length:
  - `param_pos_embed: (1, 108, d_model)`

## Model / decoder changes

- No intended functional changes relative to v3.
  - Decoder still uses UNet → latent map → patch-token transformer with `patch = 4` (256 latent tokens) and learned query tokens for parameter reconstruction.

## Training / loss changes

- No intended functional changes relative to v3.
  - v3 numerical-safety/normalization fixes (MU clamp + broadcast weights, per-scalar scaling, mean-squared hinge penalty, NCHW coercion) remain the baseline.

## Checkpoint naming

- v4 checkpoints are isolated from v3:
  - Loads/saves `Cross_CP/EPID_v4_17Dec25_checkpoint.pth` (instead of the v3 path) to avoid overwriting.

## Tooling

- Added a matching v4 inference script:
  - `inference_visualize_EPID_v4.py` (imports v4 models, defaults to v4 checkpoint/output folder).
