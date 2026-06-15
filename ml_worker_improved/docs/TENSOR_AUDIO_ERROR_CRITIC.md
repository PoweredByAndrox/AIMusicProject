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

## Build Train-Safe Mining Request

R1G locked mining slices are review-only evidence. They are not training data.
R1H converts the locked `dev`/`test` evidence into train-safe requirements that
describe what analogous rows should be mined from the `train` split.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_tensor_audio_error_train_safe_mining_request.py \
  --locked-slice-dir /app/lab_artifacts/audio_error_critic/tensor_r1_locked_mining_slice_001 \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --tensor-dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1h_train_safe_request
```

Outputs:

- `tensor_audio_error_train_safe_mining_request_manifest.json`
- `tensor_audio_error_train_safe_mining_requirements.jsonl`
- `tensor_audio_error_train_safe_mining_request_summary.md`
- `tensor_audio_error_train_safe_mining_request_validation.json`

The request keeps `locked_eval_only_source=true`, `review_only_source=true`,
`training_rows_created=false`, `training_allowed=false`, and
`release_ready=false`. It maps locked slice labels to target labels only:

- `persistent_hard_negative` and `candidate_regressed_hard_negative` request
  `hard_negative_false_accept` train examples with label `0`
- `persistent_false_reject` and `candidate_regressed_false_reject` request
  `false_reject_positive` train examples with label `1`

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_train_safe_mining_request.py \
  --request-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1h_train_safe_request
```

## Build Train-Only Mining Bundle

R1I mines only train-safe analog rows from the locked tensor dataset. It rejects
`dev`, `test`, hard-case, locked-eval, release-ready, scalar-fallback, missing
tensor-map, and leakage-key-overlap rows. Leakage overlap includes
`row_uid`, `canonical_row_key`, `candidate_id`, `track_id`, and available audio
hash/fingerprint keys. If the corpus cannot satisfy the minimum two-class train
contract, it writes
`bundle_status=blocked_insufficient_train_safe_rows` and R1J must not run.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_tensor_audio_error_train_only_mining_bundle.py \
  --request-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1h_train_safe_request \
  --tensor-dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1i_train_only_mining_bundle
```

Outputs:

- `tensor_audio_error_train_only_mining_bundle_manifest.json`
- `tensor_audio_error_train_only_training_candidates.jsonl`
- `tensor_audio_error_train_only_mining_bundle_summary.md`
- `tensor_audio_error_train_only_mining_bundle_validation.json`

Default proceed thresholds are 40 total train rows, 10 positive rows, and
10 negative rows. Usable bundles keep
`dev_test_mining_rows_used_for_training=false`,
`locked_eval_rows_used_for_training=false`,
`hardcase_rows_used_for_training=false`, `runtime_gates_changed=false`, and
`release_ready=false`.

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_train_only_mining_bundle.py \
  --bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1i_train_only_mining_bundle
```

## Train with Train-Only Mining Bundle

R1J may train a new shadow tensor critic only if R1I produced
`bundle_status=train_only_candidate_only`. The trainer accepts the R1I schema
directly and rejects R1G locked mining slice directories as training input.
It also refuses to train without locked `dev`/`test` eval rows; train rows are
not used as an eval fallback.

```bash
docker compose run --rm ml-worker python tools/train_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1j_model_003_train_safe_weighted \
  --mining-bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1i_train_only_mining_bundle \
  --mining-weight-multiplier 3.0
```

The resulting model manifest must record
`mining_bundle_schema=tensor_audio_error_train_only_mining_bundle.v1`,
`mining_non_train_rows_ignored=0`,
`dev_test_mining_rows_used_for_training=false`,
`locked_eval_rows_used_for_training=false`,
`hardcase_rows_used_for_training=false`, `shadow_only=true`, and
`release_ready=false`.

## Audit Train Negative Gap

R1L audits the fail-closed case where R1I produced train-safe positive analogs
but no usable train-safe negatives. It is artifact-only and never trains a
critic, updates runtime gates, or marks `release_ready`.

```bash
docker compose run --rm --no-deps ml-worker python tools/audit_tensor_audio_error_train_negative_gap.py \
  --tensor-dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --request-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1h_train_safe_request \
  --bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1i_train_only_mining_bundle \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1l_negative_gap_audit
```

Outputs:

- `tensor_audio_error_train_negative_gap_audit.json`
- `tensor_audio_error_train_negative_gap_rows.jsonl`
- `tensor_audio_error_train_negative_gap_summary.md`

The audit distinguishes absent negatives, missing tensor maps, leakage
exclusions, one-class train labels, and a filter/request mismatch. In the
R1H/R1I flow, R1I can block because train-safe negatives are missing from the
request outcome even when the tensor dataset still contains some train negative
rows.

## Harvest Train-Safe Hard Negatives

R1M may only emit hard-negative candidate rows from the `train` split. The
harvest artifact is not a training bundle and the trainer rejects it directly.
Source A uses existing train split tensor rows with label `0`, `train_allowed`
true, tensor maps present, no hard-case holdout flag, and no locked-eval flag.
Source B train-only generation remains blocked unless a safe train-only
generation route exists.

```bash
docker compose run --rm --no-deps ml-worker python tools/harvest_tensor_audio_error_train_hard_negatives.py \
  --negative-gap-audit-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1l_negative_gap_audit \
  --tensor-dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1m_train_hard_negative_harvest
```

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_train_hard_negative_harvest.py \
  --harvest-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1m_train_hard_negative_harvest
```

The validator fails if any row is not `split=train`, if a row is locked eval or
hard-case holdout, if `true_label` is not `0`, if tensor maps are missing, or
if any release/runtime/training safety flag is true.

## Build Balanced Train-Only Bundle

R1N merges the existing R1I positive analog bundle with the R1M hard-negative
harvest into the same train-only bundle schema accepted by the trainer. It must
contain both positive and negative train rows before R1O can run.

```bash
docker compose run --rm --no-deps ml-worker python tools/build_tensor_audio_error_balanced_train_only_bundle.py \
  --request-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1h_train_safe_request \
  --positive-bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_train_safe_loop_001/r1i_train_only_mining_bundle \
  --hard-negative-harvest-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1m_train_hard_negative_harvest \
  --tensor-dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1n_train_only_balanced_bundle
```

Proceed to R1O only when:

- `bundle_status=train_only_candidate_only`
- `candidate_rows_total >= 40`
- `positive_rows >= 10`
- `negative_rows >= 10`
- `hard_negative_rows >= 10`
- validation `ok=true`

Otherwise the flow writes a blocked report and no model artifacts should be
created.

## Train Hard-Negative Weighted Tensor Critic

R1O may train only from a usable R1N train-only bundle. It must not consume the
R1M harvest directory directly, the R1G locked mining slice, dev/test rows, or
hard-case holdout rows as fitting data.

```bash
docker compose run --rm ml-worker python tools/train_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1o_model_004_hard_negative_weighted \
  --mining-bundle-dir /app/lab_artifacts/audio_error_critic/tensor_r1_hard_negative_harvest_001/r1n_train_only_balanced_bundle \
  --mining-weight-multiplier 3.0
```

If R1N is blocked, R1O must not run.

## Evaluate Hard-Negative Variant

R1P is locked-eval only and never promotes. It can compare a trained R1O shadow
candidate against baseline and previous mining-weighted variants, but allowed
decisions remain limited to:

- `candidate_rejected_runtime`
- `keep_as_specialist_shadow_model`
- `mine_next_dataset_slice`
- `blocked_no_train_safe_hard_negatives`

It must not change runtime gates, edit the model registry, train separator
weights, train a repair adapter, publish audio, or mark `release_ready`.

## Evaluate Train-Safe Weighted Variant

R1K evaluates any R1J model only on the EVAL-1 locked split and hard-case
holdout. Use calibration, validation, scoring, shadow analysis, and locked
variant comparison as in the R1F flow. If a multi-variant comparison is needed,
baseline-vs-candidate comparisons may be run separately and summarized without
promotion.

R1K decisions are limited to:

- `candidate_rejected_runtime`
- `keep_as_specialist_shadow_model`
- `mine_next_dataset_slice`

## Decision Contract

No R1 tensor critic phase may promote a model to runtime gates, change runtime
thresholds, train separator weights, train a repair adapter, publish audio, or
mark `release_ready`. R1H uses locked evidence only as a request. R1I mines only
train rows. R1J fits only from train rows. R1K evaluates only locked
`dev`/`test` and hard-case evidence.

## Contract

`tensor_maps_required=true` and `scalar_fallback_used=false` are hard contract fields.

If training data lacks a train positive or train negative class, the trainer writes a skipped manifest with an explicit `critic_not_usable_reason` instead of training a misleading model.

Localization is diagnostic: each eval row records tensor-channel energy and `top_error_channel`. It is not a repair instruction and not a release artifact.

The manifest includes both raw 0.5-threshold metrics and a conservative `threshold_metrics` section. The conservative threshold is chosen to avoid false accepts on the locked eval split; if that forces high false rejects, the model remains shadow-only and must not be used as a runtime accept gate.
