# Tensor Audio Error Critic

R1 trains a shadow-only tensor critic/localizer from no-reference audio error maps.

The route is tensor-only:

- requires `.npz` error maps
- consumes the EVAL-1 locked split
- does not use scalar MODEL-2C fallback features
- does not train separator weights
- does not train a repair adapter
- does not change runtime gates or thresholds
- never marks `release_ready`

## Build Tensor Dataset

```bash
docker compose run --rm ml-worker python tools/build_locked_tensor_audio_error_dataset.py \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001
```

Outputs:

- `locked_tensor_audio_error_dataset_manifest.json`
- `tensor_audio_error_dataset.jsonl`
- `tensor_audio_error_skipped_pairs.jsonl`
- `tensor_maps/*.npz`
- `tensor_dataset_progress.jsonl`

The builder fails if `leakage_report.json` or `leak_safe_split_manifest.json` reports critical leakage.

## Train Shadow Critic

```bash
docker compose run --rm ml-worker python tools/train_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001
```

Outputs:

- `tensor_audio_error_critic_manifest.json`
- `tensor_audio_error_critic.joblib`
- `tensor_audio_error_critic_eval.jsonl`
- `tensor_audio_error_localization.jsonl`
- `tensor_audio_error_critic_summary.json`

The trainer evaluates multiple tensor-only candidate classifiers and selects
the best shadow model by conservative zero-false-accept coverage first, then
AUROC and accuracy. Current candidate kinds are:

- `logistic_regression_balanced`
- `random_forest_balanced`

The selected `model_kind` and all `candidate_model_metrics` are recorded in
`tensor_audio_error_critic_manifest.json`.

## Calibrate Shadow Thresholds

```bash
docker compose run --rm --no-deps ml-worker python tools/calibrate_tensor_audio_error_critic.py \
  --model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_calibration_001 \
  --max-false-accept-rate 0.0
```

Outputs:

- `tensor_audio_error_thresholds.json`
- `tensor_audio_error_threshold_sweep.jsonl`

Calibration reads only the locked eval predictions from the trained shadow
critic. It refuses `release_ready` models and scalar-fallback models, and it
marks the calibration `not_usable` if no positive eval coverage is possible
within the false-accept budget.

Validation is fail-closed for calibration artifacts: the calibration schema,
`tensor_maps_required`, `shadow_only`, and recorded `model_dir` must match the
validated model artifact.

## Validate

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_critic.py \
  --model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001 \
  --calibration-dir /app/lab_artifacts/audio_error_critic/tensor_r1_calibration_001 \
  --require-usable-calibration \
  --require-trained
```

## Score Shadow Rows

```bash
docker compose run --rm --no-deps ml-worker python tools/score_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001 \
  --calibration-dir /app/lab_artifacts/audio_error_critic/tensor_r1_calibration_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_scores_001
```

Outputs:

- `tensor_audio_error_shadow_scores.jsonl`
- `tensor_audio_error_shadow_score_summary.json`

Shadow scores use the calibrated threshold only for diagnostic
`shadow_decision` fields. They do not update runtime gates and every row keeps
`runtime_gate_decision_applied=false`, `shadow_only=true`, and
`release_ready=false`.

## Analyze Shadow Scores

```bash
docker compose run --rm --no-deps ml-worker python tools/analyze_tensor_audio_error_shadow_scores.py \
  --scores-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_scores_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_analysis_001
```

Outputs:

- `tensor_audio_error_shadow_analysis_summary.json`
- `tensor_audio_error_hard_negative_candidates.jsonl`
- `tensor_audio_error_false_reject_candidates.jsonl`
- `tensor_audio_error_model_improvement_report.md`

The analyzer is diagnostic only. It identifies hard negatives and false
rejects for future dataset/model improvement planning, while keeping
`training_started=false`, `runtime_gates_changed=false`, and
`release_ready=false`.

## Build Mining Bundle

```bash
docker compose run --rm --no-deps ml-worker python tools/build_tensor_audio_error_mining_bundle.py \
  --analysis-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_analysis_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_mining_bundle_001
```

Outputs:

- `tensor_audio_error_mining_bundle_manifest.json`
- `tensor_audio_error_training_candidates.jsonl`
- `tensor_audio_error_mining_bundle_summary.md`

The mining bundle is `candidate_only`: it preserves hard-negative and
false-reject provenance for future training review, but it does not train a
critic, change runtime gates, or mark anything `release_ready`.
The builder fails closed if shadow analysis safety flags are violated or if
input rows do not match the expected hard-negative/false-reject provenance.

## Contract

`tensor_maps_required=true` and `scalar_fallback_used=false` are hard contract fields.

If training data lacks a train positive or train negative class, the trainer writes a skipped manifest with an explicit `critic_not_usable_reason` instead of training a misleading model.

Localization is diagnostic: each eval row records tensor-channel energy and `top_error_channel`. It is not a repair instruction and not a release artifact.

The manifest includes both raw 0.5-threshold metrics and a conservative `threshold_metrics` section. The conservative threshold is chosen to avoid false accepts on the locked eval split; if that forces high false rejects, the model remains shadow-only and must not be used as a runtime accept gate.
