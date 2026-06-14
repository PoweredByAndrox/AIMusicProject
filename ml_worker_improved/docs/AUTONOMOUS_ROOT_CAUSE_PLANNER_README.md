# Autonomous Root-Cause Planner

`tools/plan_autonomous_root_causes.py` consumes the autonomous decision queue and turns blocked or
hard-case decisions into deterministic next-step planning buckets.

This tool is planner-only. It does not copy/export audio, execute repair, start fine-tuning, change
policy, request new model assets, add weights, add cleanup/debleed/denoise stages, or relax gates.

## Inputs

- `autonomous_decision_queue.jsonl`
- `autonomous_judge_manifest.json`
- `hard_case_benchmark.jsonl`
- `control_plane_dashboard.json`
- `golden_corpus.jsonl`

The benchmark file is used to preserve original track-level split and hard-case metadata. Golden
rows are used only as protected holdout exclusions and never for training or threshold tuning.

## Outputs

- `root_cause_plan.jsonl`
- `root_cause_plan_summary.md`
- `root_cause_plan_manifest.json`

Each output row includes:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `original_decision`
- `dominant_failure_reason`
- `failure_labels`
- `judge_mode`
- `confidence`
- `root_cause_bucket`
- `safe_next_action`
- `forbidden_actions`
- `required_evidence_before_execution`
- `can_execute_automatically`
- `can_enter_finetune_pool`
- `can_change_policy`
- `can_request_new_model`
- `human_review_required`

## Bucket Policy

- `candidate_space_exhausted` stays a stop/planning signal and maps to
  `candidate_space_exhausted` unless non-protected strong evidence supports
  `current_model_insufficient`.
- `no_accepted_candidate` maps to `no_accepted_candidate` or, under strong non-protected evidence,
  `current_model_insufficient`.
- `unknown_blocked_reason` maps to `unknown_needs_more_evidence` unless reference-aware evidence
  indicates `reference_or_alignment_problem`.
- holdout or golden track hashes map to `protected_holdout_only` before training/tuning buckets.
- minus rows are routed to `minus_specific_failure`, never `vocal_specific_failure`.
- vocal issue rows can route to `vocal_specific_failure` or `policy_classification_problem`.

## Safety

Every planned row has:

- `can_execute_automatically=false`
- `human_review_required=false`

The manifest safety flags remain false for:

- `audio_copied`
- `training_started`
- `policy_changed`
- `models_changed`
- `weights_added`
- `cleanup_routes_added`
- `thresholds_changed`

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_autonomous_root_causes.py \
  --autonomous-decisions-jsonl /app/lab_artifacts/control_plane_phase7_20260528/autonomous_decision_queue.jsonl \
  --autonomous-manifest-json /app/lab_artifacts/control_plane_phase7_20260528/autonomous_judge_manifest.json \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --dashboard-json /app/lab_artifacts/control_plane_phase2_20260528/control_plane_dashboard.json \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase8_20260528
```
