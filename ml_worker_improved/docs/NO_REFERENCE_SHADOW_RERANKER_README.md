# No-Reference Shadow Reranker

`tools/run_no_reference_shadow_reranker.py` integrates the local weak-label
critic as a shadow reranker for existing no-reference candidates. It compares
script-only, learned-only, and hybrid decisions without changing optimizer
output or publishing a result.

## Contract

- Learned critic is shadow/diagnostic only.
- Learned accept never bypasses script blockers or non-accept script decisions.
- Final guarded decisions remain blocked/rejected when script blockers exist.
- Optimizer rows are attached as context only and are not overwritten.
- No audio is rendered, copied, exported, or decoded.
- No separator training, fine-tune, model registry edit, threshold relaxation,
  cleanup route, or new separator model is performed.
- All output rows keep `can_publish_as_result=false`,
  `can_mark_release_ready=false`, `release_ready=false`, and
  `delivery_allowed=false`.

## Inputs

```text
/app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl
/app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl
/app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl
/app/lab_artifacts/no_reference_ml_critic/models/model_manifest.json
/app/lab_artifacts/no_reference_ml_critic/models/vocal_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/minus_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/pair_critic.joblib
/app/lab_artifacts/no_reference_ml_critic/models/policy_controller.joblib
```

The candidate pool enriches windowed judge rows with provenance. Windowed judge
rows remain the per-candidate script source of truth. Optimizer rows are
track-level context only.

## Outputs

```text
/app/lab_artifacts/no_reference_ml_critic/shadow/shadow_reranker_result.jsonl
/app/lab_artifacts/no_reference_ml_critic/shadow/shadow_reranker_summary.md
/app/lab_artifacts/no_reference_ml_critic/shadow/disagreement_cases.jsonl
```

Each candidate row includes:

- `track_id`, `candidate_id`, `score_mode`
- `script_score`, `learned_score`, `hybrid_score`
- `script_decision`, `learned_decision`, `hybrid_decision`
- `final_guarded_decision`
- `script_blockers`, `learned_blocker_probability`
- `safety_override_applied`
- `disagreement_type`
- `can_publish_as_result=false`
- `can_mark_release_ready=false`

`hybrid_score = 0.5 * script_score + 0.5 * learned_score`.

## Disagreements

The route writes disagreement cases for:

- `script_accept_learned_reject`
- `script_reject_learned_accept`
- `rank_order_disagreement`

Each disagreement includes script reason, learned confidence, dominant defect,
blocker labels, and one recommended next action:

- `inspect_with_window_judge`
- `add_to_hard_case_pool`
- `candidate_for_preview_render`
- `keep_script_decision`

## Docker Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_run_no_reference_shadow_reranker.py \
  tests/test_train_no_reference_critic.py \
  tests/test_build_no_reference_ml_dataset.py -q
```

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_shadow_reranker.py \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --optimizer-result-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --output-jsonl /app/lab_artifacts/no_reference_ml_critic/shadow/shadow_reranker_result.jsonl \
  --summary-md /app/lab_artifacts/no_reference_ml_critic/shadow/shadow_reranker_summary.md \
  --disagreements-jsonl /app/lab_artifacts/no_reference_ml_critic/shadow/disagreement_cases.jsonl
```

The runner logs `acceleration_mode=external_only_cpu_metadata_scoring` and
`gpu_inference_skipped=true` because this phase scores existing metadata with
local sklearn critics only; it does not run neural separation or audio export.
