# Fine-tune Candidate Preflight

`tools/plan_finetune_candidate_preflight.py` performs a metadata-only preflight over Phase 8
`can_enter_finetune_pool=true` rows. It classifies which rows have enough manifest evidence for a
future dataset-construction phase.

This phase does not add models, add weights, start training, build datasets, build ground-truth
bundles, copy/export audio, add cleanup/debleed/denoise stages, change policy, or relax thresholds.

## Inputs

- `root_cause_plan.jsonl`
- `root_cause_plan_manifest.json`
- `autonomous_decision_queue.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`
- `reference_path_recovery_plan.jsonl`
- `reference_path_recovery_manifest.json`

Only root-cause rows with `can_enter_finetune_pool=true` are included.

## Outputs

- `finetune_candidate_preflight.jsonl`
- `finetune_candidate_preflight_manifest.json`
- `finetune_candidate_preflight_summary.md`

Each row includes source/track metadata, root-cause provenance, manifest evidence booleans, a strict
`preflight_status`, `required_before_training`, `forbidden_actions`, and hard `can_start_training_now=false`.

## Status Policy

- `blocked_holdout`: split is `holdout`.
- `blocked_golden`: track hash exists in protected `golden_corpus.jsonl`.
- `blocked_policy_issue`: root cause remains a policy/taxonomy issue.
- `blocked_missing_reference`: no verified reference path or Phase 10F explicitly blocked reference recovery.
- `blocked_missing_source_audio`: no source audio path is present in metadata.
- `blocked_missing_gt_bundle`: no verified ground-truth bundle manifest is attached.
- `blocked_insufficient_evidence`: reference/source/GT exist, but alignment/residual/leakage evidence or stem type is incomplete.
- `eligible_metadata_only`: metadata is sufficient for a later dataset-construction proposal, but training is still forbidden.

## Safety

The planner reads JSON/JSONL manifests only. It may check path existence for metadata paths, but it
does not read audio bytes, hash audio, copy audio, create datasets, build GT bundles, run metrics, or
start training.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_finetune_candidate_preflight.py \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --root-cause-manifest-json /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan_manifest.json \
  --autonomous-decisions-jsonl /app/lab_artifacts/control_plane_phase7_20260528/autonomous_decision_queue.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --reference-path-recovery-jsonl /app/lab_artifacts/control_plane_phase10f_20260528/reference_path_recovery_plan.jsonl \
  --reference-path-recovery-manifest-json /app/lab_artifacts/control_plane_phase10f_20260528/reference_path_recovery_manifest.json \
  --output-dir /app/lab_artifacts/control_plane_phase11_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_finetune_candidate_preflight.py \
  tests/test_plan_autonomous_root_causes.py \
  tests/test_artifact_status_contract.py \
  -q
```
