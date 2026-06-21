# ARIS Train >90 Cycle 14 Plan

## Source-of-truth checkpoint

- Run: `vocal_shared_adapter_matrix_v13_strong_scnet_melband_holdout_ncw_v17`
- Decision artifact: `train_ready_decision_vocal_matrix_v13_strong_scnet_melband_ncw_v17/train_ready_audio_model_decision_manifest.json`
- Decision: `train_result_not_ready`
- Success rate: `0.0`
- Train ready: `false`
- Release ready: `false`

## Confirmed failure signals

1. `strong_family_magnitude_identity_guard` completed training and wrote a checkpoint, but holdout failed:
   - holdout rows: `0/2` passed
   - failed check: `improvement_below_threshold`
   - holdout improvements: `-0.497437 dB`, `-0.500223 dB`
   - baseline SDR remained stronger than the adapted candidate.
2. `strong_family_magnitude_detail_repair` did not train:
   - failed phase: `training`
   - error: `CUDA free memory is below required minimum`
   - progress event: `cuda_memory_guard_blocked`
   - no CPU fallback was used.

## Ranked intervention ideas

### 1. Matrix CUDA phase cleanup

Hypothesis: the second config failed because the same Python process retained CUDA allocator state after config 1 train/holdout. A fail-closed cleanup phase between train, holdout, and configs should allow the next config to start on GPU or produce a cleaner blocker.

Minimum experiment:
- add explicit CUDA synchronize / empty-cache / IPC cleanup between matrix phases;
- log cleanup start/finish events in matrix progress;
- rerun the same matrix or at least the blocked config.

Success criterion:
- config 2 reaches `dataset_train_loop_start` or fails with a non-leftover blocker;
- no CPU fallback;
- release_ready remains false.

### 2. No-harm adapter variant

Hypothesis: the current adapter is destructive on an already strong holdout source. A future config should learn or apply a conservative no-harm residual/gating strategy instead of always replacing the source with the adapted estimate.

Minimum experiment:
- only after cleanup fix, add a separate config/route with explicit no-harm objective and holdout improvement gate;
- do not relax thresholds;
- do not mark pass unless holdout improvement is positive and above threshold.

Success criterion:
- holdout improvement becomes positive on both NCW holdout candidates;
- success rate is strictly above `0.90`.

### 3. Verified corpus balance expansion

Hypothesis: two training track ids are too narrow for a production adapter and overfit Square Oxygen / one short Hotstepper example.

Minimum experiment:
- expand verified vocal training rows from project-owned/golden sources only;
- exclude dirty no-reference stems as ground truth;
- rerun locked holdout.

Success criterion:
- no leakage;
- min unique training tracks increases;
- holdout success improves without threshold relaxation.

## Execution order

1. Implement and test CUDA cleanup in `run_generic_audio_finetune_matrix.py`.
2. Rerun a GPU matrix iteration using the same locked NCW holdout, starting with the previously blocked config.
3. Build train-ready decision with `--min-success-rate 0.90`.
4. If success remains `<= 0.90`, continue to no-harm adapter variant or verified corpus expansion.

## Safety contract

- No fallback.
- No threshold relaxation.
- No runtime gate bypass.
- No release_ready.
- No model registry edit.
- Dirty stems are not ground truth.
