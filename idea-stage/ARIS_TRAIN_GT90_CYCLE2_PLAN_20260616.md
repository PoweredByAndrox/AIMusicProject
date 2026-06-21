# ARIS Train >90 Cycle 2 Plan

## Source-of-truth checkpoint

Cycle 1 added and validated three safety layers:

- `tools/validate_verified_audio_training_corpus.py`
- `tools/build_project_owned_audio_permission_decisions.py`
- `tools/build_train_ready_audio_model_decision.py`

Docker validation:

- targeted fine-tune/corpus suite: `61 passed`
- attestation suite after human-feedback opt-in fix: `4 passed`

Runtime artifacts:

- corpus: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/corpus`
- recovery queue: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/audio_training_recovery`
- merged verified corpus v3: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/merged_audio_training_v3_human_optin`
- verified corpus manifest: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/verified_corpus_v3_human_optin/verified_audio_training_corpus_manifest.json`
- vocal matrix: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v1_human_optin`
- train-ready decision: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/train_ready_decision_vocal_matrix_v1`

Confirmed facts:

- verified trainable examples: `55`
- overall unique tracks: `4`
- vocal tracks: `3`
- matrix GPU route ran with `device=cuda`
- matrix status: `failed`
- best config: none
- best observed vocal holdout SDR on `nothing-can-wait`: `0.157136 dB`
- best observed improvement over source: `+7.931508 dB`
- train-ready decision: `train_result_not_ready`
- success_rate: `0.0`
- release_ready: `false`

## Failure interpretation

The failure is not a threshold issue and not a CPU/GPU routing issue.
The failed checks are objective holdout checks:

- `holdout_failed`
- `holdout_failed_evaluations`
- `holdout_row_failed`
- `holdout_metric_below_threshold`

The current shared teacher-mask adapter improves over source but does not reach
the absolute stem-reference quality gate on cross-track holdout.

## Cycle 2 TOP actions

1. Run leave-one-track-out vocal folds for the other available vocal holdouts.
   Keep the existing thresholds and Docker/CUDA route. This tests whether the
   current failure is fold-specific or general.

2. Add an aggregate leave-one-track-out decision manifest if multiple fold
   matrix outputs exist. The aggregate decision must require success rate
   strictly greater than `0.90`, no leakage, no release-ready, and no runtime
   gate changes.

3. If all folds fail, stop spending cycles on this weak adapter shape and move
   to a stronger model/training route in Cycle 3. Candidate direction:
   verified-corpus teacher-student separator fine-tune or repair adapter that
   learns spectrogram residual errors rather than a shallow mask-only adapter.

## Hard constraints

- No threshold relaxation.
- No release-ready.
- No default runtime promotion.
- No dirty stems as ground truth.
- No fallback/degraded route marked as success.
- Docker/CUDA required for training and holdout inference.
