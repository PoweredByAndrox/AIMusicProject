# Audio Error Critic Baseline

MODEL-2A trains a lightweight, shadow-only audio error critic from the repaired
MODEL-1 audio error grounding dataset. It scores candidate windows; it does not
modify audio, repair stems, train a separator, change runtime gates, publish a
result, or mark anything `release_ready`.

## Inputs

The trainer reads only the repaired strict-provenance dataset:

- `repaired_audio_error_grounding_manifest.json`
- `repaired_audio_error_window_dataset.jsonl`
- `repaired_audio_error_maps_index.jsonl`
- `provenance_repair_report.json`
- `error_maps/*.npz` when present

The original mix is still used only through MODEL-1 mixture-consistency
features. Unsafe or unrepairable rows are never used as supervised labels.

## Strict Training Filter

A row can be used only when all of these are true:

- `safe_for_training=true`
- `release_ready=false`
- real `candidate_id`
- real `route_id`, not `unknown`
- real `pair_compatibility_status`
- real `judge_decision`
- valid window boundaries
- existing `mix_path`, `vocal_candidate_path`, and `minus_candidate_path`
- an existing error-map `.npz` or complete scalar error features

Forbidden rows are counted in `excluded_reason_counts`. Synthetic candidate ids,
missing route ids, `unknown` route ids, unsafe rows, release-ready rows, and
rows with missing audio are excluded.

## Model

Default model:

- sklearn `RandomForestClassifier` for accept/block probability
- sklearn `RandomForestRegressor` heads for localized severity scores
- `DictVectorizer` for deterministic scalar/categorical feature encoding

No fallback model is produced. If sklearn is unavailable, safe rows are below
the configured minimum, or the positive/negative class is missing, the trainer
writes diagnostics and does not write `audio_error_critic_model.joblib`.

For `safe_rows >= 100` and `< 500`, the model is allowed as a baseline but
`model_quality_confidence=low`. It remains shadow-only and cannot promote to a
production gate.

## Outputs

Training output directory:

- `audio_error_critic_model.joblib` only when training is valid
- `audio_error_critic_manifest.json`
- `audio_error_critic_metrics.json`
- `audio_error_critic_eval.jsonl`
- `audio_error_critic_summary.md`
- `scored_audio_error_windows.jsonl`
- `audio_error_critic_progress.jsonl`

Evaluation output directory:

- `audio_error_critic_metrics.json`
- `audio_error_critic_eval.jsonl`
- `audio_error_critic_summary.md`

Scoring output directory:

- `scored_audio_error_windows.jsonl`
- `audio_error_critic_score_summary.json`

## Safety

All MODEL-2A artifacts keep:

- `unsafe_rows_used=false`
- `shadow_only=true`
- `separator_training_started=false`
- `release_ready=false`
- `melody_transcription_ran=false`
- `recommendation_index_updated=false`
- `can_promote_to_production_gate=false`

The critic cannot bypass script blockers or existing runtime safety gates.
MODEL-2B may integrate it only as shadow scoring.

## Commands

Tests:

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_train_audio_error_critic.py \
  tests/test_evaluate_audio_error_critic.py \
  tests/test_score_audio_error_windows_with_critic.py \
  tests/test_validate_audio_error_critic_model.py \
  -q
```

Train:

```bash
docker compose run --rm ml-worker python tools/train_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001 \
  --min-safe-rows 100 \
  --model-kind random_forest \
  --group-split track_or_candidate \
  --shadow-only true
```

Evaluate:

```bash
docker compose run --rm --no-deps ml-worker python tools/evaluate_audio_error_critic.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001 \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001_eval
```

Score:

```bash
docker compose run --rm --no-deps ml-worker python tools/score_audio_error_windows_with_critic.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001 \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired \
  --output-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001_scores
```

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_audio_error_critic_model.py \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001 \
  --eval-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001_eval \
  --scores-dir /app/lab_artifacts/audio_error_critic/model_2a_baseline_001_scores
```
