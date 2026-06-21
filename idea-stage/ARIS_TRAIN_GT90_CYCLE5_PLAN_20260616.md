# ARIS TRAIN GT90 Cycle 5 Plan

## Checkpoint

V5 completed with structured progress and no active `ml-worker` left running.
All four balanced/projection configs trained on CUDA and failed the locked
`nothing-can-wait` vocal holdout because candidate SDR stayed below
`min_sdr_db=3.0`.

## Root Cause Hypothesis

The current shared teacher-mask adapter only predicts magnitude masks and
reuses source phase during application. V5 improved the source baseline by about
7 dB but stayed below absolute SDR, which is consistent with unresolved
phase/complex reconstruction error rather than a pure routing or threshold
issue.

## Next Route

Add and evaluate `mask_output_mode=complex` in the existing teacher-mask adapter:

- train a bounded complex ratio mask from verified source/target STFT;
- include complex STFT reconstruction loss;
- keep application reference-free;
- reject source-feature memory with complex mode to avoid reconstructive memory
  fallback;
- keep delivery blocked until holdout gate passes.

## V6 Matrix

Run three configs:

- `complex_one_per_track`
- `complex_two_per_track`
- `complex_wide_context`

All configs use `source_feature_memory_neighbors=0`, `device=cuda`, locked
holdout track `nothing-can-wait`, and unchanged thresholds:
`min_sdr_db=3.0`, `min_improvement_db=0.5`, `min_checkpoint_pass_rate=1.0`.

## Safety

- fallback_used=false
- release_ready=false
- delivery_allowed=false
- thresholds_relaxed=false
- model_registry_edited=false
- no new separator model
- no dirty stems as ground truth
