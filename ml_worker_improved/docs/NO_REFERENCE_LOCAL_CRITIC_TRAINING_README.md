# No-Reference Local Critic Training

This route builds a local weak-label critic/ranker from existing no-reference
script artifacts. It is metadata-only during dataset build and does not train
or change separator models.

## Contract

- Input labels come from existing no-reference decisions, not reference stems.
- Generated stems are never treated as ground truth.
- Dataset split unit is `track_id`.
- Rows without explicit `score_mode` are marked as weak legacy component-proxy
  examples with `strict_mode_label=false`.
- Learned models are hybrid rankers only. Script safety blockers override
  learned accept decisions.
- Outputs remain diagnostic/model-routing artifacts and never set
  `release_ready=true`.

## Files

- Dataset builder: `tools/build_no_reference_ml_dataset.py`
- Trainer: `tools/train_no_reference_critic.py`
- Evaluator: `tools/eval_no_reference_critic.py`
- Inference wrapper: `tools/run_no_reference_learned_critic.py`
- Tests:
  - `tests/test_build_no_reference_ml_dataset.py`
  - `tests/test_train_no_reference_critic.py`

## Outputs

```text
/app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_ml_examples.jsonl
/app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_pairwise_examples.jsonl
/app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_ml_dataset_manifest.json
/app/lab_artifacts/no_reference_ml_critic/models/vocal_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/minus_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/pair_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/policy_controller.joblib
/app/lab_artifacts/no_reference_ml_critic/models/model_manifest.json
/app/lab_artifacts/no_reference_ml_critic/eval/evaluation_summary.md
```

## Docker Run

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_build_no_reference_ml_dataset.py \
  tests/test_train_no_reference_critic.py -q
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_ml_dataset.py \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --optimizer-result-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl \
  --segment-blend-plan-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl \
  --segment-blend-preview-window-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl \
  --output-dir /app/lab_artifacts/no_reference_ml_critic/dataset
```

```bash
docker compose run --rm --no-deps ml-worker python tools/train_no_reference_critic.py \
  --examples-jsonl /app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_ml_examples.jsonl \
  --pairwise-jsonl /app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_pairwise_examples.jsonl \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models
```

```bash
docker compose run --rm --no-deps ml-worker python tools/eval_no_reference_critic.py \
  --examples-jsonl /app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_ml_examples.jsonl \
  --pairwise-jsonl /app/lab_artifacts/no_reference_ml_critic/dataset/no_reference_pairwise_examples.jsonl \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --output-dir /app/lab_artifacts/no_reference_ml_critic/eval
```

## Safety

The learned inference wrapper emits:

- `learned_vocal_score`
- `learned_minus_score`
- `learned_pair_score`
- `learned_policy_action`
- `learned_confidence`
- `script_score`
- `hybrid_score`

`hybrid_score = 0.5 * script_score + 0.5 * learned_score`.

If script blockers, severe reasons, blocked decisions, or forbidden actions are
present, the wrapper sets `learned_policy_action=block_by_safety_gate`,
`delivery_allowed=false`, and `release_ready=false`.
