# Target Audio Error Critics

MODEL-2C trains and audits two shadow-only target critics:

- `minus_accept_block_classifier`
- `vocal_accept_block_classifier`

The critics use only target-specific minus/vocal training views. They do not
read pair critic data, overall accept data, unsafe rows, unrepairable rows, or
final-status-only labels. Separator models, runtime gates, cleanup routes,
melody transcription, recommendation indexes, downloads, and `release_ready`
remain unchanged.

## Inputs

Required views directory:

- `audio_error_training_views_manifest.json`
- `minus_critic_training_view.jsonl`
- `vocal_critic_training_view.jsonl`
- `audio_error_training_views_summary.md`

Training reads only `minus_critic_training_view.jsonl` and
`vocal_critic_training_view.jsonl`. Pair and overall views may exist in the
directory, but they are not loaded as training labels.

## Strict Training Filter

A row can train a target critic only when:

- `safe_for_this_view=true`
- `view_name` matches `minus_critic` or `vocal_critic`
- `target_name` is either the view name or the stem target (`minus` / `vocal`)
- `candidate_id` is real and not synthetic
- `route_id` is real and not `unknown`
- `candidate_audio_path` and `mix_path` exist
- `label` is `0` or `1`
- `label_source` is the exact target windowed judge file
- `release_ready=false`

Unsafe, unrepairable, pair, overall, and final-status-only rows are not labels.

Accepted no-reference minus listen-preview rows with
`label_source=no_reference_minus_windowed_judge.jsonl` are score-only evidence:
they may be scored by an already trained shadow critic to localize current
candidate windows, but they do not train the critic because they are not exact
target windowed judge labels.

The training-view manifest mirrors that split: `positive_labels` and
`negative_labels` include all safe rows in the view, while
`training_positive_labels`, `training_negative_labels`, and
`training_safe_rows` count only strict rows that the trainer can load.
`can_train_minus_critic` and `can_train_vocal_critic` are based on those strict
training counts, not on score-only rows.

RALPH minus improvement hardcase rows imported through
`build_audio_error_training_views.py --ralph-manifests` use the same score-only
label source. They preserve RALPH residual/detail-drift provenance for shadow
scoring and router diagnosis, but they are excluded from strict target critic
training for the same reason.

## Reliability Controls

MODEL-2C makes false accept the primary safety metric.

- Default split is `candidate_id`, with split audit metadata written per target.
- `target_audio_error_critic_thresholds.json` stores the selected accept/block
  thresholds.
- Threshold selection is false-accept-first: it must satisfy
  `false_accept_rate <= --max-false-accept-rate` before optimizing recall.
- The accept boundary is strict: `predicted_accept_probability` must be greater
  than `accept_threshold`; equality remains in the abstain/block region so a
  boundary score is never promoted to `accept_candidate`.
- Abstain is allowed for shadow scoring and is never counted as accept.
- Hard-negative weighting applies only to real negative rows with configured
  blocker labels.
- A critic is shadow-usable only when reliability gates pass.

Required validator gates:

- `unsafe_rows_used=false`
- `pair_rows_used=false`
- `overall_rows_used=false`
- `shadow_only=true`
- `release_ready=false`
- `candidate_id_overlap_train_test=false`
- `test_negative_rows >= --min-negative-test-rows`
- `false_accept_rate <= --max-false-accept-rate`
- calibration is either applied or has an explicit skipped reason
- per-route and per-track false-accept audits exist

## Calibration

`tools/calibrate_target_audio_error_critics.py` calibrates probabilities from
already trained target critics. It does not train separators.

Supported methods:

- `auto`
- `sigmoid`
- `isotonic`

If dev data is insufficient or single-class, calibration is skipped with an
explicit reason.

Calibration output:

- `critic_calibration_manifest.json`
- `critic_calibration_metrics.json`
- `critic_calibration_summary.md`
- `calibrated_minus_audio_error_critic.joblib` when calibrated
- `calibrated_vocal_audio_error_critic.joblib` when calibrated

## Outputs

Training output:

- `target_audio_error_critics_manifest.json`
- `target_audio_error_critic_thresholds.json`
- `target_audio_error_critic_reliability_report.json`
- `minus_audio_error_critic.joblib`
- `vocal_audio_error_critic.joblib`
- `minus_audio_error_critic_metrics.json`
- `vocal_audio_error_critic_metrics.json`
- `target_audio_error_critic_eval.jsonl`
- `target_audio_error_critic_summary.md`

Evaluation output:

- `target_audio_error_critic_eval_summary.json`
- `minus_audio_error_critic_metrics.json`
- `vocal_audio_error_critic_metrics.json`
- `target_audio_error_critic_eval.jsonl`
- `target_audio_error_critic_summary.md`

Scoring output:

- `scored_minus_critic_windows.jsonl`
- `scored_vocal_critic_windows.jsonl`
- `target_audio_error_critic_score_summary.json`

Score-only rows are labeled diagnostic evidence, not training labels. The
scorer preserves `label_source` and `score_only` on each scored window and writes
a `score_only_audit` section in `target_audio_error_critic_score_summary.json`.
If a score-only negative row is predicted as `accept_candidate`, the summary sets
`score_only_blind_spot_detected=true` and the target status to
`blocked_model_blind_spot`. This is a shadow-scoring failure signal for router
diagnosis; it does not change thresholds, promote diagnostic rows, or mark any
artifact as listen-ready.

All artifacts keep `shadow_only=true`, `pair_rows_used=false`,
`overall_rows_used=false`, `unsafe_rows_used=false`,
`separator_training_started=false`, and `release_ready=false`.

## Commands

Tests:

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_calibrate_target_audio_error_critics.py \
  tests/test_train_target_audio_error_critics.py \
  tests/test_evaluate_target_audio_error_critics.py \
  tests/test_validate_target_audio_error_critics.py \
  -q
```

Train:

```bash
docker compose run --rm ml-worker python tools/train_target_audio_error_critics.py \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --targets minus,vocal \
  --model-kind random_forest \
  --group-split candidate_id \
  --class-weight balanced \
  --negative-class-weight 3.0 \
  --hard-negative-blockers severe_phase_collapse,severe_vocal_bleed_in_minus,severe_phase_comb_filtering \
  --threshold-objective false_accept_first \
  --max-false-accept-rate 0.10 \
  --min-negative-test-rows 20 \
  --enable-abstain-zone true \
  --abstain-margin 0.10 \
  --shadow-only true
```

Calibrate:

```bash
docker compose run --rm --no-deps ml-worker python tools/calibrate_target_audio_error_critics.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_calibration \
  --method auto
```

Evaluate:

```bash
docker compose run --rm --no-deps ml-worker python tools/evaluate_target_audio_error_critics.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --calibration-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_calibration \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_eval \
  --use-thresholds true
```

Score:

```bash
docker compose run --rm --no-deps ml-worker python tools/score_target_audio_error_views.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --calibration-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_calibration \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_scores \
  --use-thresholds true \
  --allow-abstain true
```

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_target_audio_error_critics.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --calibration-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_calibration \
  --eval-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_eval \
  --scores-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001_scores \
  --max-false-accept-rate 0.10 \
  --min-negative-test-rows 20
```
