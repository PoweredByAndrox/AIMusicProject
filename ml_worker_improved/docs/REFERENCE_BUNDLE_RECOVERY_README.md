# Reference Bundle Recovery Plan

`tools/plan_reference_bundle_recovery.py` consumes Phase 10B collected evidence and determines
whether missing reference or ground-truth inputs can be recovered later.

This phase is plan-only. It does not copy/export audio, build a ground-truth bundle, promote
artifacts, start training, change policy, change thresholds, add models, add weights, or add
cleanup/debleed/denoise stages.

## Inputs

- `collected_evidence.jsonl`
- `collected_evidence_manifest.json`
- `failed_evidence_collection.jsonl`
- `missing_evidence_plan.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`

Only collected Phase 10B rows are recovery-planned. Rows in `failed_evidence_collection.jsonl` are
preserved as skipped/blocked source-of-truth context and are not turned into recovery execution.

## Outputs

- `reference_bundle_recovery_plan.jsonl`
- `reference_bundle_recovery_manifest.json`
- `reference_bundle_recovery_summary.md`

Each output row includes:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `source_manifest`
- `current_missing_evidence`
- `discovered_audio_inputs`
- `discovered_reference_inputs`
- `discovered_manifest_inputs`
- `required_bundle_inputs`
- `recovery_status`
- `can_build_gt_bundle`
- `can_run_alignment_check`
- `can_run_reference_metrics`
- `can_run_residual_metrics`
- `can_run_leakage_metrics`
- `required_next_tool`
- `blocked_reason`
- `forbidden_actions`

## Recovery Policy

- `recoverable`: source mix exists and either a target reference exists for comparison or FL Studio
  stem exports exist for a future internal ground-truth bundle.
- `partially_recoverable`: source manifest/source mix exists, but generated target or reference
  inputs are incomplete.
- `blocked_missing_source_audio`: source manifest exists but the original/source mix is missing.
- `blocked_missing_reference`: generated target exists but no target reference or rebuildable FL
  Studio export is present.
- `blocked_missing_manifest`: source manifest is missing or unreadable.
- `not_recoverable`: insufficient provenance remains after the explicit blockers above.

## Safety

The planner records existing paths only. It never creates a bundle and never writes audio. Holdout
and golden rows receive explicit forbidden actions so future tooling cannot route them into training
or threshold tuning.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_reference_bundle_recovery.py \
  --collected-evidence-jsonl /app/lab_artifacts/control_plane_phase10b_20260528/collected_evidence.jsonl \
  --collected-evidence-manifest-json /app/lab_artifacts/control_plane_phase10b_20260528/collected_evidence_manifest.json \
  --failed-evidence-jsonl /app/lab_artifacts/control_plane_phase10b_20260528/failed_evidence_collection.jsonl \
  --missing-evidence-plan-jsonl /app/lab_artifacts/control_plane_phase10a_20260528/missing_evidence_plan.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10c_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_reference_bundle_recovery.py \
  tests/test_run_missing_evidence_collection.py \
  tests/test_artifact_status_contract.py \
  -q
```
