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

Optional mining-bundle weighting can be used for a follow-up shadow training
run. Only mining rows with `split=train` can affect `sample_weight`; `dev` and
`test` mining rows are counted as ignored holdout evidence and are never used
for fitting.
The trainer rejects mining bundles with unsafe flags, unknown split names, or
labels that disagree with matched tensor training rows.

```bash
docker compose run --rm ml-worker python tools/train_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_002_mining_weighted \
  --mining-bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_mining_bundle_001 \
  --mining-weight-multiplier 3.0
```

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

## Compare Shadow Variants on Locked Eval

R1F compares tensor critic variants only on the EVAL-1 locked split and its
locked hard-case holdout. The comparator is decision/report only: it does not
train, promote a model, change runtime gates, train separator weights, train a
repair adapter, or mark `release_ready`.

```bash
docker compose run --rm --no-deps ml-worker python tools/compare_tensor_audio_error_critic_variants.py \
  --locked-eval-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --baseline-model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001 \
  --baseline-scores-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_scores_001 \
  --baseline-analysis-dir /app/lab_artifacts/audio_error_critic/tensor_r1_shadow_analysis_001 \
  --candidate-model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_002_mining_weighted \
  --candidate-scores-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_002_mining_weighted_scores \
  --candidate-analysis-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_002_mining_weighted_analysis \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_variant_decision_001
```

Outputs:

- `tensor_audio_error_critic_variant_comparison.json`
- `tensor_audio_error_critic_variant_comparison.jsonl`
- `tensor_audio_error_critic_decision_manifest.json`
- `tensor_audio_error_critic_decision_summary.md`

The comparison records AUROC, accuracy, hard-negative count, false-reject
count, false-accept rate, false-reject rate, accept coverage, hard-case holdout
metrics, per-stem metrics, per-route metrics, per-track metrics, calibration
status, mining rows used, and locked-split contamination status.
Decision metrics are computed from locked eval rows only (`dev`/`test`);
`train` score rows and full shadow-analysis counts are retained as audit
metadata and are not allowed to drive the variant decision.

Decision values are intentionally limited to:

- `keep_as_specialist_shadow_model`
- `mine_next_dataset_slice`
- `candidate_rejected_runtime`

`promote` is not a valid R1F decision. A variant with hard-negative or
false-reject improvements but no hard-case holdout improvement remains shadow
evidence and should feed the next locked mining slice instead of changing
runtime gates.

Validate the decision artifact:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_critic_decision.py \
  --decision-dir /app/lab_artifacts/audio_error_critic/tensor_r1_variant_decision_001
```

The validator fails if `release_ready=true`, runtime gates changed, separator
or repair-adapter training started, a non-locked split was used, dev/test
mining rows were used for training, `comparison.locked_eval.errors` is
non-empty, baseline/candidate `safety_errors` is non-empty, the decision
attempts promotion, or hard-case holdout metrics are missing.

## Build Locked Mining Slice from Variant Decision

When the R1F decision is `mine_next_dataset_slice`, the next safe artifact is a
locked-eval-only mining slice. This is not the same as a train mining bundle:
it preserves `dev`/`test` and hard-case evidence for review, explicitly ignores
`train` score rows, and sets `training_allowed=false`.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_tensor_audio_error_locked_mining_slice.py \
  --decision-dir /app/lab_artifacts/audio_error_critic/tensor_r1_variant_decision_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_locked_mining_slice_001
```

Outputs:

- `tensor_audio_error_locked_mining_slice_manifest.json`
- `tensor_audio_error_locked_mining_slice.jsonl`
- `tensor_audio_error_locked_mining_slice_summary.md`

The slice labels locked eval rows as:

- `candidate_regressed_false_reject`
- `persistent_false_reject`
- `candidate_regressed_hard_negative`
- `persistent_hard_negative`

Validate the slice:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_locked_mining_slice.py \
  --slice-dir /app/lab_artifacts/audio_error_critic/tensor_r1_locked_mining_slice_001
```

The builder fails before writing a slice if the source decision was
`candidate_rejected_runtime`, `comparison.locked_eval.errors` is non-empty, or
baseline/candidate `safety_errors` is non-empty. The validator fails if the
slice includes train rows, marks training allowed, changes runtime gates,
starts training, starts separator/repair training, or marks `release_ready`.

## Contract

`tensor_maps_required=true` and `scalar_fallback_used=false` are hard contract fields.

If training data lacks a train positive or train negative class, the trainer writes a skipped manifest with an explicit `critic_not_usable_reason` instead of training a misleading model.

Localization is diagnostic: each eval row records tensor-channel energy and `top_error_channel`. It is not a repair instruction and not a release artifact.

The manifest includes both raw 0.5-threshold metrics and a conservative `threshold_metrics` section. The conservative threshold is chosen to avoid false accepts on the locked eval split; if that forces high false rejects, the model remains shadow-only and must not be used as a runtime accept gate.
