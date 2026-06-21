# ARIS Train >90 Cycle 11 Plan

## Checkpoint

- v10 all-verified `htdemucs_ft` source-family matrix failed.
- Best v10 improvement was only `+0.002030 dB`, far below `+0.5 dB`.
- Source-family audit against verified NCW target:
  - `htdemucs_ft_v2`: `12.254840 dB`
  - `htdemucs_6s`: `11.893627 dB`
  - `scnet_xl_ihf_musdb18`: `16.130864 dB`
  - `mdx23c_vocals`: `75.928163 dB`
  - `melband_roformer_vocals`: `16.153894 dB`

## Reconciliation

The NCW verified target is aligned with the MDX23C-style vocal target. Using
`mdx23c_vocals` as dirty source would be identity leakage and offers no repair
learning signal. `scnet` and `melband` are already stronger than `htdemucs_ft`
against the same target. `htdemucs_6s` is the only audited family with clear
repair headroom and explicit CUDA route.

## Chosen Next Experiment

Generate `htdemucs_6s` candidate vocal sources for all 8 verified vocal rows,
build a full candidate-conditioned training view, and run v11 matrix with
unchanged holdout thresholds.

## Contract

- Existing registered `htdemucs_6s` model only.
- No download.
- No threshold relaxation.
- No model registry edit.
- Dirty candidate is source only.
- Verified target remains target/holdout reference.
- No fallback to raw mix.
- No release-ready.
- Docker-only execution with progress artifacts.
