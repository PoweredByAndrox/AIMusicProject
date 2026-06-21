# ARIS Train GT90 Cycle 13 Plan

Date: 2026-06-16

## Source-of-Truth Checkpoint

- Active goal: reach verified TRAIN success >90% on reproducible Docker/ARIS cycle.
- Latest matrix: `vocal_shared_adapter_matrix_v12_multi_family_htd6_ft_holdout_ncw`.
- Latest decision: `train_result_not_ready`, `success_rate=0.0`, `train_ready=false`.
- Safety remains closed: `release_ready=false`, `thresholds_relaxed=false`, `runtime_gates_changed=false`, `model_registry_edited=false`.

## Confirmed Failure

v12 trained two CUDA configs over htdemucs_6s + htdemucs_ft candidate-conditioned sources.
Both configs failed holdout:

- `multi_family_magnitude_identity_guard`: `holdout_failed`, `holdout_metric_below_threshold`.
- `multi_family_magnitude_balanced_detail`: `holdout_failed`, `holdout_metric_below_threshold`.

The failure is not a logging/GPU blocker: progress artifacts show CUDA training through epoch 180/180 and matrix finish.

## ARIS Brainstorm Result

Do not repeat adapter-only htd6/ft training with more epochs. Training loss decreased but locked holdout did not pass.

The next route should change the source family signal:

1. Generate candidate-conditioned vocal sources for the same verified examples using stronger non-target-like families:
   - `scnet_xl_ihf_musdb18`
   - `melband_roformer_vocals`
2. Exclude `mdx23c_vocals` from training for this cycle because NCW audit shows it is target-like against the current target (`baseline_sdr_db=75.928163`), which can mask failure as identity.
3. Build a four-family view:
   - `htd6`
   - `ft`
   - `scnet`
   - `melband`
4. Run a Docker-only CUDA matrix with unchanged gates:
   - `min_sdr_db=3.0`
   - `min_improvement_db=0.5`
   - `min_checkpoint_pass_rate=1.0`
5. Build train-ready decision with `min_success_rate=0.90`.

## Execution Contract

- No fallback.
- No dirty stem as ground truth.
- Candidate source is source-only, never target.
- Verified target path is preserved.
- Docker-only execution.
- GPU required for model candidate generation and training.
- No threshold relaxation.
- No registry edits.
- No `release_ready`.

## Stop / Continue Gate

Stop this cycle only if the decision artifact shows:

- `train_ready=true`
- `success_rate > 0.90`
- `critical_leakage_count=0`
- `thresholds_relaxed=false`
- `runtime_gates_changed=false`
- `release_ready=false`

Otherwise continue into the next ARIS cycle.
